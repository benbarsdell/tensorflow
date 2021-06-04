/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/permutation_input_iterator.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename Tindex, typename Tsegmentids>
__global__ void SegmentOffsetsKernel(
    Tindex size, Tsegmentids nsegments,
    const Tsegmentids* __restrict__ segment_ids,  // [size]
    Tindex* __restrict__ segment_offsets) {       // [nsegments + 1]
  GPU_1D_KERNEL_LOOP(i, size + 1) {
    // IDs are clipped to [-1, nsegments] so that out-of-bounds IDs are ignored.
    // Note that we can't report invalid IDs from the GPU without incurring
    // additional overhead.
    auto clip = [&](Tsegmentids id) {
      return min(max(Tsegmentids(-1), id), nsegments);
    };
    const Tsegmentids cur_id = (i < size) ? clip(segment_ids[i]) : nsegments;
    const Tsegmentids prev_id =
        (i == 0) ? Tsegmentids(-1) : clip(segment_ids[i - 1]);
    // At segment boundaries, write the offset for this ID and any missing IDs
    // since the previous one.
    for (Tsegmentids id = prev_id + 1; id <= cur_id; ++id) {
      segment_offsets[id] = i;
    }
  }
}

// Finds the start offset of each segment in the given sorted segment_ids
// vector. Missing IDs are given the same offset as the next ID so that they
// represent empty ranges. Invalid IDs (those that are outside the range
// [0, nsegments)) are ignored. The value at segment_offsets[0] is set to the
// start index of the first valid ID (e.g., 0 if all IDs are valid), and the
// value at segment_offsets[nsegments] is set to the end index of the last valid
// ID (e.g., nsegments if all IDs are valid).
template <typename Tindex, typename Tsegmentids>
Status LaunchSegmentOffsetsKernel(const GPUDevice& d, Tindex size,
                                  Tsegmentids nsegments,
                                  const Tsegmentids* segment_ids,  // [size]
                                  Tindex* segment_offsets) {  // [nsegments + 1]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      size + 1, d, &SegmentOffsetsKernel<Tindex, Tsegmentids>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentOffsetsKernel<Tindex, Tsegmentids>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), size, nsegments, segment_ids,
                         segment_offsets);
}

template <typename T>
struct RealTypeIfComplex {
  using type = T;
};

template <typename Real>
struct RealTypeIfComplex<std::complex<Real>> {
  using type = Real;
};

// Reduces along columns of the thread block, returning the result in the first
// row of threads.
template <typename T, typename ReduceOp>
__device__ T ReduceBlockAlongCols(ReduceOp reduce_op, const T& value,
                                  bool is_valid) {
  GPU_DYNAMIC_SHARED_MEM_DECL(/*ALIGN=*/16, char, shared_memory_raw);
  T* const shared_partial_reduction =
      reinterpret_cast<T*>(shared_memory_raw);  // [blockDim.y, blockDim.x]
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  T reduced = value;
  // Reduce over the y dimension of the block.
  for (unsigned k = blockDim.y / 2; k > 0; k /= 2) {
    if (is_valid && y < 2 * k) {
      shared_partial_reduction[y * blockDim.x + x] = reduced;
    }
    __syncthreads();
    if (is_valid && y < k) {
      reduced = reduce_op(reduced,
                          shared_partial_reduction[(y + k) * blockDim.x + x]);
    }
    __syncthreads();
  }
  return reduced;
}

// This kernel uses a 2D thread decomposition. The x dimension maps to the inner
// dimension of the input/output. The y grid dimension maps to segments, and y
// threads within a block cooperate to reduce over the block's segment.
// Note that Tinit is needed because Tvec and Treducevec may be vector types,
// but Tinit is always a scalar type.
// Note that the first dimension of input_vec is nouter if indices is not
// provided; otherwise it is indexed indirectly via indices and can have any
// size (as long as it spans at least the maximum value in indices).
template <typename Treducevec, typename Tvec, typename Tindex,
          typename Tsegmentids, typename ReduceOp, typename Tinit>
__global__ void SegmentReduceVectorKernel(
    Tindex nouter, Tindex ninner_vec, Tsegmentids nsegments, ReduceOp reduce_op,
    Tinit initial_value, Tinit empty_segment_value, bool is_mean, bool is_sqrtn,
    const Tvec* __restrict__ input_vec,          // [nouter or any, ninner_vec]
    const Tindex* __restrict__ segment_offsets,  // [nsegments + 1]
    const Tindex* __restrict__ indices,          // [nouter] (optional)
    Tvec* __restrict__ output_vec) {             // [nsegments, ninner_vec]
  const int num_blocks_x = (ninner_vec - 1) / blockDim.x + 1;
  // Grid-stride loop over inner dimension blocks.
  for (Tindex blk_x = blockIdx.x; blk_x < num_blocks_x; blk_x += gridDim.x) {
    const Tindex x = threadIdx.x + blk_x * blockDim.x;
    const Tindex y = threadIdx.y;
    const bool x_ok = x < ninner_vec;
    // Grid-stride loop over segment blocks, each processing one segment.
    for (Tsegmentids seg = blockIdx.y; seg < nsegments; seg += gridDim.y) {
      // Load segment range.
      const Tindex begin = segment_offsets[seg];
      const Tindex end = segment_offsets[seg + 1];
      // Reduce over the segment.
      Treducevec result = Treducevec(initial_value);
      // Loop over the segment, reducing blockDim.y elements at a time.
      for (Tindex y_offset = begin; y_offset < end; y_offset += blockDim.y) {
        const bool y_ok = (y_offset + y) < end;
        // Perform indirect lookup if required.
        const Tindex y_idx =
            indices && y_ok ? indices[y_offset + y] : y_offset + y;
        const int64 input_idx = static_cast<int64>(y_idx) * ninner_vec + x;
        // Load the input row from global mem.
        Treducevec block_result =
            x_ok && y_ok ? input_vec[input_idx] : Tvec(initial_value);
        // Reduce along the columns of the block, returning result in first row.
        block_result = ReduceBlockAlongCols(reduce_op, block_result, x_ok);
        if (y == 0 && x_ok) {
          result = reduce_op(result, block_result);
        }
      }
      // First row of the block stores the result to global memory.
      if (y == 0 && x_ok) {
        if (begin == end) {
          // Empty segment.
          result = Treducevec(empty_segment_value);
        } else {
          typename RealTypeIfComplex<Tinit>::type total_weight(end - begin);
          // Normalize the results if necessary.
          if (is_mean) {
            result /= Treducevec(total_weight);
          } else if (is_sqrtn) {
            result /= Treducevec(sqrt(total_weight));
          }
        }
        // Cast from Treducevec to Tvec.
        const int64 output_idx = static_cast<int64>(seg) * ninner_vec + x;
        output_vec[output_idx] = static_cast<Tvec>(result);
      }
    }
  }
}

// Reduces input matrix within segments over the outer dimension. Empty segments
// always output empty_segment_value.
// If is_mean or is_sqrtn is true, the results are normalized using the
// corresponding function.
// If indices is not nullptr, input rows are accessed indirectly as
// input[indices[i]], instead of input[i].
// Note: Treducevec is to allow reducing in higher precision than Tvec.
template <typename Treducevec, typename Tvec, typename Tindex,
          typename Tsegmentids, typename ReduceOp, typename Tinit>
Status LaunchSegmentReduceVectorKernel(
    const GPUDevice& d, Tindex nouter, Tindex ninner_vec, Tsegmentids nsegments,
    ReduceOp reduce_op, Tinit initial_value, Tinit empty_segment_value,
    bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,          // [nouter or any, ninner_vec]
    const Tindex* segment_offsets,  // [nsegments + 1]
    const Tindex* indices,          // [nouter] (optional)
    Tvec* output_vec) {             // [nsegments, ninner_vec]
  static constexpr const int kMaxGridX = (1u << 31) - 1;
  static constexpr const int kMaxGridY = (1u << 16) - 1;
  const int max_block_size = 1024;  // Can be tuned for perf (<= 1024)
  const int min_block_size = 64;    // Can be tuned for perf
  const Tindex ninner_pow2 = Tindex(1) << Log2Ceiling64(ninner_vec);
  // This is a heuristic that first allocates threads in the block to the inner
  // (x) dimension (which is most efficient) and then allocates the rest to the
  // reduction (y) dimension (which is less efficient but increases
  // parallelism).
  int block_x = std::min(ninner_pow2, static_cast<Tindex>(max_block_size));
  const Tindex avg_reduce_size =
      Eigen::divup(nouter, static_cast<Tindex>(nsegments));
  const Tindex avg_reduce_size_pow2 = Tindex(1)
                                      << Log2Ceiling64(avg_reduce_size);
  dim3 block(
      block_x,
      std::min(static_cast<Tindex>(Eigen::divup(min_block_size, block_x)),
               avg_reduce_size_pow2));
  dim3 grid(std::min(Eigen::divup(ninner_vec, static_cast<Tindex>(block.x)),
                     static_cast<Tindex>(kMaxGridX)),
            std::min(nsegments, static_cast<Tsegmentids>(kMaxGridY)));
  unsigned shared_memory_bytes = block.x * block.y * sizeof(Treducevec);
  return GpuLaunchKernel(
      SegmentReduceVectorKernel<Treducevec, Tvec, Tindex, Tsegmentids, ReduceOp,
                                Tinit>,
      grid, block, shared_memory_bytes, d.stream(), nouter, ninner_vec,
      nsegments, reduce_op, initial_value, empty_segment_value, is_mean,
      is_sqrtn, input_vec, segment_offsets, indices, output_vec);
}

template <typename Tvec, typename Treducevec, typename Tindex,
          typename Tsegmentids, typename Tinit>
__global__ void SegmentReduceEpilogueKernel(
    Tsegmentids nsegments, Tinit empty_segment_value, bool is_mean,
    bool is_sqrtn,
    const Treducevec* __restrict__ output_raw,   // [nsegments]
    const Tindex* __restrict__ segment_offsets,  // [nsegments + 1]
    Tvec* __restrict__ output) {                 // [nsegments]
  GPU_1D_KERNEL_LOOP(seg, nsegments) {
    Tindex segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    Treducevec val = output_raw[seg];
    if (segment_size == 0) {
      // Empty segment.
      val = Treducevec(empty_segment_value);
    } else if (is_mean) {
      val /= Treducevec(segment_size);
    } else if (is_sqrtn) {
      val /= Treducevec(
          sqrt(typename RealTypeIfComplex<Tinit>::type(segment_size)));
    }
    // Cast from Treducevec to Tvec.
    output[seg] = static_cast<Tvec>(val);
  }
}

// Normalizes output_raw based on segment size and casts from Treducevec to
// Tvec. If Tvec == Treducevec, this is safe to call with output_raw == output.
// Note that Treducevec is the type that was used for the reduction, which may
// be a higher-precision type than the output type Tvec (e.g., float vs. half).
template <typename Tvec, typename Treducevec, typename Tindex,
          typename Tsegmentids, typename Tinit>
Status LaunchSegmentReduceEpilogueKernel(
    const GPUDevice& d, Tsegmentids nsegments, Tinit empty_segment_value,
    bool is_mean, bool is_sqrtn,
    const Treducevec* output_raw,   // [nsegments]
    const Tindex* segment_offsets,  // [nsegments + 1]
    Tvec* output) {                 // [nsegments]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      nsegments, d,
      &SegmentReduceEpilogueKernel<Tvec, Treducevec, Tindex, Tsegmentids,
                                   Tinit>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(
      SegmentReduceEpilogueKernel<Tvec, Treducevec, Tindex, Tsegmentids, Tinit>,
      config.block_count, config.thread_per_block, 0, d.stream(), nsegments,
      empty_segment_value, is_mean, is_sqrtn, output_raw, segment_offsets,
      output);
}

template <typename Tto>
struct CastFunctor {
  template <typename T>
  __device__ Tto operator()(const T& val) const {
    return static_cast<Tto>(val);
  }
};

template <typename Treducevec, typename Tvec, typename Tindex,
          typename Tsegmentids, typename ReduceOp, typename Tinit>
Status SegmentReduceGPUImplNoInnerDim(
    OpKernelContext* ctx, Tindex nouter,
    Tsegmentids nsegments, ReduceOp reduce_op, Tinit initial_value,
    Tinit empty_segment_value, bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,          // [nouter or any]
    const Tindex* segment_offsets,  // [nsegments + 1]
    const Tindex* indices,          // [nouter] (optional)
    Tvec* output_vec) {             // [nsegments]
  // Here we use gpuprim::DeviceSegmentedReduce (which is optimized for this
  // shape) and add the additional required functionality using fancy input
  // iterators and an epilogue kernel.

  // Note: This reinterpret cast is only needed to avoid compilation error
  // when Tvec != Treducevec; the result is only used if Tvec == Treducevec.
  Treducevec* output_raw_ptr = reinterpret_cast<Treducevec*>(output_vec);
  Tensor output_raw;
  bool need_temp_output = !std::is_same<Tvec, Treducevec>::value;
  if (need_temp_output) {
    // Note: We must allocate and reinterpret as bytes because Treducevec may
    // be a vector type and they are not supported as Tensor dtypes.
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8,
        TensorShape({static_cast<int64>(nsegments * sizeof(Treducevec))}),
        &output_raw));
    output_raw_ptr =
        reinterpret_cast<Treducevec*>(output_raw.flat<int8>().data());
  }
  if (indices) {
    // Use fancy iterators to do lookup via indices and to cast to Treducevec.
    PermutationInputIterator<Treducevec, decltype(input_vec),
                             decltype(indices)>
        perm_iter(input_vec, indices);
    gpuprim::TransformInputIterator<Treducevec, CastFunctor<Treducevec>,
                                    decltype(perm_iter)>
        input_lookup_cast(perm_iter, {});
    TF_RETURN_IF_ERROR(
        GpuSegmentedReduce(ctx, nsegments, reduce_op, Treducevec(initial_value),
                           input_lookup_cast, segment_offsets, output_raw_ptr));
  } else {
    // Use a fancy iterator to cast to Treducevec.
    gpuprim::TransformInputIterator<Treducevec, CastFunctor<Treducevec>,
                                    decltype(input_vec)>
        input_cast(input_vec, {});
    TF_RETURN_IF_ERROR(GpuSegmentedReduce(ctx, nsegments, reduce_op,
                                          Treducevec(initial_value), input_cast,
                                          segment_offsets, output_raw_ptr));
  }
  bool need_epilogue = !std::is_same<Tvec, Treducevec>::value ||
                       initial_value != empty_segment_value || is_mean ||
                       is_sqrtn;
  if (need_epilogue) {
    const GPUDevice& device = ctx->eigen_gpu_device();
    // Normalize based on the segment size and cast results back to T.
    TF_RETURN_IF_ERROR(LaunchSegmentReduceEpilogueKernel(
        device, nsegments, empty_segment_value, is_mean, is_sqrtn,
        output_raw_ptr, segment_offsets, output_vec));
  }
  return Status::OK();
}

template <typename Treducevec, typename Tvec, typename Tindex,
          typename Tsegmentids, typename ReduceOp, typename Tinit>
Status SegmentReduceGPUImpl(
    OpKernelContext* ctx, Tindex nouter, Tindex ninner_vec,
    Tsegmentids nsegments, ReduceOp reduce_op, Tinit initial_value,
    Tinit empty_segment_value, bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,           // [nouter or any, ninner_vec]
    const Tsegmentids* segment_ids,  // [nouter]
    const Tindex* indices,           // [nouter] (optional)
    Tvec* output_vec) {              // [nsegments, ninner_vec]
  const GPUDevice& device = ctx->eigen_gpu_device();

  if (nouter == 0) {
    // Just set output to empty_segment_value.
    GPUDevice d = ctx->template eigen_device<GPUDevice>();
    int64 output_size = static_cast<int64>(nsegments) * ninner_vec;
    GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d);
    return GpuLaunchKernel(SetToValue<Tvec, Tinit>, config.block_count,
                           config.thread_per_block, 0, d.stream(), output_size,
                           output_vec, empty_segment_value);
  }

  // Allocate and compute segment_offsets.
  Tensor segment_offsets;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<Tindex>::value,
                                        TensorShape({nsegments + 1}),
                                        &segment_offsets));
  Tindex* segment_offsets_ptr = segment_offsets.flat<Tindex>().data();
  TF_RETURN_IF_ERROR(LaunchSegmentOffsetsKernel(
      device, nouter, nsegments, segment_ids, segment_offsets_ptr));

  const Tindex avg_reduce_size =
      Eigen::divup(nouter, static_cast<Tindex>(nsegments));
  // This avg_reduce_size threshold is a performance heuristic.
  if (ninner_vec == 1 && avg_reduce_size >= 512) {
    // Here we use a gpuprim-based implementation that doesn't support an
    // inner dimension but can be significantly faster for large reductions.
    return SegmentReduceGPUImplNoInnerDim<Treducevec>(
        ctx, nouter, nsegments, reduce_op, initial_value, empty_segment_value,
        is_mean, is_sqrtn, input_vec, segment_offsets_ptr, indices, output_vec);
  }
  // Here we use a custom kernel that is optimized for ninner_vec >= ~64 and
  // gives decent performance for smaller cases. It also handles indices,
  // casting to/from Treducevec, and normalizing the output.
  return LaunchSegmentReduceVectorKernel<Treducevec>(
      device, nouter, ninner_vec, nsegments, reduce_op, initial_value,
      empty_segment_value, is_mean, is_sqrtn, input_vec, segment_offsets_ptr,
      indices, output_vec);
}

template <typename Treduce>
struct SegmentReduceGPUVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T, typename Tindex, typename Tsegmentids,
              typename ReduceOp>
    Status operator()(OpKernelContext* ctx, Tindex nouter, Tindex ninner,
                      Tsegmentids nsegments, ReduceOp reduce_op,
                      T initial_value, T empty_segment_value, bool is_mean,
                      bool is_sqrtn, const T* input,
                      const Tsegmentids* segment_ids, const Tindex* indices,
                      T* output) {
      DCHECK_EQ(ninner % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(input) % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(output) % vec_size, 0);
      Tindex ninner_vec = ninner / vec_size;
      using Tvec = AlignedVector<T, vec_size>;
      using Treducevec = AlignedVector<Treduce, vec_size>;
      const Tvec* input_vec = reinterpret_cast<const Tvec*>(input);
      Tvec* output_vec = reinterpret_cast<Tvec*>(output);

      return SegmentReduceGPUImpl<Treducevec>(
          ctx, nouter, ninner_vec, nsegments, reduce_op, initial_value,
          empty_segment_value, is_mean, is_sqrtn, input_vec, segment_ids,
          indices, output_vec);
    }
  };
};

// Reduces input matrix within segments over the outer dimension. Empty segments
// always output empty_segment_value.
// The segment_ids vector must be sorted.
// If is_mean or is_sqrtn is true, the results are normalized using the
// corresponding function.
// If indices is not nullptr, input rows are accessed indirectly as
// input[indices[i]], instead of input[i].
// The implementation is deterministic.
// Note: Treduce is to allow reducing in higher precision than T.
template <typename Treduce, typename T, typename Tindex, typename Tsegmentids,
          typename ReduceOp>
Status SegmentReduceGPU(OpKernelContext* ctx, Tindex nouter, Tindex ninner,
                        Tsegmentids nsegments, ReduceOp reduce_op,
                        T initial_value, T empty_segment_value, bool is_mean,
                        bool is_sqrtn,
                        const T* input,  // [nouter or any, ninner]
                        const Tsegmentids* segment_ids,  // [nouter]
                        const Tindex* indices,           // [nouter] (optional)
                        T* output) {                     // [nsegments, ninner]
  if (ninner == 0 || nsegments == 0) return Status::OK();
  return DispatchToVectorized<
      T, SegmentReduceGPUVectorized<Treduce>::template Impl>(
      MinAlignmentOf(input, output, ninner), ctx, nouter, ninner, nsegments,
      reduce_op, initial_value, empty_segment_value, is_mean, is_sqrtn, input,
      segment_ids, indices, output);
}

template <typename ReduceOp, typename T>
struct ReduceType {
  using type = T;
};

// Sum fp16 values using an fp32 accumulator to avoid numerical issues.
template <>
struct ReduceType<functor::Sum, Eigen::half> {
  using type = float;
};

namespace functor {

template <typename T, typename Index, typename InitialValueF,
          typename EmptySegmentValueF, typename ReductionF>
void SegmentReductionFunctor<
    T, Index, InitialValueF, EmptySegmentValueF,
    ReductionF>::operator()(OpKernelContext* ctx, const GPUDevice& d,
                            const Index output_rows,
                            const TensorShape& segment_ids_shape, bool is_mean,
                            typename TTypes<Index>::ConstFlat segment_ids,
                            const Index data_size, const T* data,
                            typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }

  // Launch kernel(s) to compute sorted segment reduction.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;
  const Index num_segments = output.size() / input_inner_dim_size;
  using Treduce = typename ReduceType<ReductionF, T>::type;
  TF_CHECK_OK(SegmentReduceGPU<Treduce>(
      ctx, input_outer_dim_size, input_inner_dim_size, num_segments,
      ReductionF(), InitialValueF()(), EmptySegmentValueF()(),
      /*is_mean=*/is_mean, /*is_sqrtn=*/false, data, segment_ids.data(),
      /*indices = */ static_cast<const Index*>(nullptr), output.data()));
}

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat unsorted_segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }
    // Launch kernel(s) to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = unsorted_segment_ids.dimension(0);
    const Index input_inner_dim_size = data.dimension(1);
    const Index num_segments = output.size() / input_inner_dim_size;
    using Treduce = typename ReduceType<ReductionF, T>::type;

    // Allocate temporary space and sort segment_ids, then call the sorted implem.
    Tensor segment_ids;
    TF_CHECK_OK(ctx->allocate_temp(
        DataTypeToEnum<Index>::value,
        TensorShape({static_cast<int64>(input_outer_dim_size)}), &segment_ids));
    Index* segment_ids_ptr = segment_ids.flat<Index>().data();
    Tensor sorted_indices;
    TF_CHECK_OK(ctx->allocate_temp(
        DataTypeToEnum<Index>::value,
        TensorShape({static_cast<int64>(input_outer_dim_size)}),
        &sorted_indices));
    Index* sorted_indices_ptr = sorted_indices.flat<Index>().data();
    TF_CHECK_OK(GpuRadixSort(
        ctx, input_outer_dim_size, unsorted_segment_ids.data(), segment_ids_ptr,
        /*indices=*/static_cast<const Index*>(nullptr), sorted_indices_ptr));
    TF_CHECK_OK(SegmentReduceGPU<Treduce>(
        ctx, input_outer_dim_size, input_inner_dim_size, num_segments,
        ReductionF(), /*initial_value=*/InitialValueF()(),
        /*empty_segment_value=*/InitialValueF()(), /*is_mean=*/false,
        /*is_sqrtn=*/false, /*input=*/data.data(),
        /*segment_ids=*/segment_ids_ptr, /*indices=*/sorted_indices_ptr,
        output.data()));
  }
};

template <typename T, typename Index, typename SegmentId>
Status SparseSegmentReductionFunctor<T, Index, SegmentId>::operator()(
    OpKernelContext* context, bool is_mean, bool is_sqrtn, T default_value,
    typename TTypes<T, 2>::ConstTensor input,
    typename TTypes<Index>::ConstVec indices,
    typename TTypes<SegmentId>::ConstVec segment_ids,
    typename TTypes<T, 2>::Tensor output) {
  using ReduceOp = functor::Sum;
  using Treduce = typename ReduceType<ReduceOp, T>::type;
  Index nouter = segment_ids.size();
  Index ninner = input.dimension(1);
  SegmentId nsegments = output.dimension(0);
  return SegmentReduceGPU<Treduce>(
      context, /*nouter=*/nouter, /*ninner=*/ninner,
      /*nsegments=*/nsegments, /*reduce_op=*/ReduceOp(),
      /*initial_value=*/T(0),
      /*empty_segment_value=*/default_value,
      /*is_mean=*/is_mean, /*is_sqrtn=*/is_sqrtn,
      /*input=*/input.data(), /*segment_ids=*/segment_ids.data(),
      /*indices=*/indices.data(), /*output=*/output.data());
}

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index)                            \
  template struct SegmentReductionFunctor<T, Index, functor::Zero<T>,      \
                                          functor::Zero<T>, functor::Sum>; \
  template struct SegmentReductionFunctor<T, Index, functor::One<T>,       \
                                          functor::One<T>, functor::Prod>; \
  template struct SegmentReductionFunctor<T, Index, functor::Highest<T>,   \
                                          functor::Zero<T>, functor::Min>; \
  template struct SegmentReductionFunctor<T, Index, functor::Lowest<T>,    \
                                          functor::Zero<T>, functor::Max>;

#define DEFINE_SORTED_GPU_SPECS(T)         \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Lowest<T>, functor::Max>;    \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index,                  \
                                         functor::Highest<T>, functor::Min>;   \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::Prod>;

// Sum is the only op that supports all input types currently.
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index)         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, \
                                         functor::Zero<T>, functor::Sum>;

#define DEFINE_REAL_GPU_SPECS(T)                  \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int64);

#define DEFINE_SUM_GPU_SPECS(T)                  \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SUM_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DEFINE_SUM_GPU_SPECS);

#undef DEFINE_SORTED_GPU_SPECS_INDEX
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

// TODO(benbarsdell): These kernels are disabled on Windows as a workaround for
// a CI build error: "formal parameter with requested alignment of 128 won't be
// aligned". The root cause is suspected to be an aligned type (AlignedVector)
// being passed to a function by value, possibly inside the CUB library
// somewhere, but I have not yet been able to reproduce it in isolation outside
// of the GitHub CI.
#if !defined(PLATFORM_WINDOWS)

#define DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR(T)                \
  template struct SparseSegmentReductionFunctor<T, int32, int32>; \
  template struct SparseSegmentReductionFunctor<T, int32, int64>; \
  template struct SparseSegmentReductionFunctor<T, int64, int32>; \
  template struct SparseSegmentReductionFunctor<T, int64, int64>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR);
#undef DEFINE_SPARSE_SEGMENT_REDUCTION_FUNCTOR

#endif  // !defined(PLATFORM_WINDOWS)

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
