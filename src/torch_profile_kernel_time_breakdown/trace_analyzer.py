from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
from research_mono.utils.pydantic_utils import MyBaseModel

_dir_server2local_data = Path("/Users/tom/temp/temp_sglang_server2local")


class TracePerScenarioInfo(MyBaseModel):
    layer_indicator_kernel: str
    layer_indicator_kernel_sparsity: int
    category_info: Dict[str, List[str]]
    kernels_to_add_index: List[str] = []


class TracePerFileInfo(MyBaseModel):
    trace_name: str
    interest_time_ranges: List[Tuple[int, int]]
    time_multiplier: float = 1.0


def analyze_trace(per_file_info_name: str):
    return analyze_trace_raw(
        pack_name=per_file_info_name,
        per_file_info=PER_FILE_INFOS[per_file_info_name],
        per_scenario_info=PER_SCENARIO_INFOS[per_file_info_name.split("/")[0]],
    )


def analyze_trace_raw(
    pack_name: str,
    per_file_info: TracePerFileInfo,
    per_scenario_info: TracePerScenarioInfo,
):
    layer_indicator_kernel_sparsity = per_scenario_info.layer_indicator_kernel_sparsity

    df = pl.read_json(_dir_server2local_data / per_file_info.trace_name)
    df = df.select("traceEvents").explode("traceEvents").unnest("traceEvents")
    df = df.filter(pl.col("ph") == "X")
    assert len(per_file_info.interest_time_ranges) == 1
    interest_time_range = per_file_info.interest_time_ranges[0]
    df = df.filter(
        (pl.col("ts") >= interest_time_range[0] / 1000)
        & (pl.col("ts") < interest_time_range[1] / 1000)
    )
    # df = df.filter(reduce(
    #     lambda a, b: a | b,
    #     [(pl.col('ts') >= start / 1000) & (pl.col('ts') < end / 1000) for start, end in per_file_info.interest_time_ranges]
    # ))
    df = df.filter(pl.col("cat") == "kernel")

    df = df.sort("ts")
    assert df[0, "name"] == per_scenario_info.layer_indicator_kernel

    df_sub = df.filter(pl.col("name") == per_scenario_info.layer_indicator_kernel)
    num_remove_indicator_kernel = (len(df_sub) + 1) % layer_indicator_kernel_sparsity
    print(f"{pack_name=} {per_file_info.trace_name=} {num_remove_indicator_kernel=}")
    if num_remove_indicator_kernel == 1:
        df = df.filter(pl.col("ts") < df_sub[-1, "ts"])
    elif num_remove_indicator_kernel == 0:
        pass
    else:
        raise NotImplementedError
    assert (
        len(df.filter(pl.col("name") == per_scenario_info.layer_indicator_kernel)) + 1
    ) % layer_indicator_kernel_sparsity == 0

    df = df.with_columns(
        section_index=(
            (pl.col("name") == per_scenario_info.layer_indicator_kernel).cum_sum() - 1
        )
        // layer_indicator_kernel_sparsity,
    )
    df = df.with_columns(
        name_index_in_section=pl.col("name").cum_count().over(["section_index", "name"])
        - 1,
    )
    df = df.with_columns(
        name=pl.when(pl.col("name").is_in(per_scenario_info.kernels_to_add_index))
        .then(pl.col("name_index_in_section").cast(str) + "#" + pl.col("name"))
        .otherwise(pl.col("name")),
    )

    category_of_name = {
        name: category
        for category, names in per_scenario_info.category_info.items()
        for name in names
    }
    df = df.with_columns(
        category=pl.col("name").replace_strict(
            category_of_name, default="Other Kernels"
        ),
    )
    df = _compute_bubble(df)
    df_raw = df
    print(f"{df_raw['section_index'].max()=} {df_raw['section_index'].min()=}")

    df = df_raw.group_by("category", "name")
    df = df.agg(
        count_kernel=pl.len(),
        dur_mean=pl.col("dur").mean(),
        dur_std=pl.col("dur").std(),
    )
    df = df.sort("dur_mean", descending=True)
    df_agg_by_name = df

    df = df_raw.filter(pl.col("name") == per_scenario_info.layer_indicator_kernel)
    df = df[::layer_indicator_kernel_sparsity]
    df = df.with_columns(layer_dur=pl.col("ts").diff()).filter(
        pl.col("layer_dur").is_not_null()
    )
    df = df["layer_dur"]
    row_overall = dict(
        category="Overall", dur_mean=df.mean(), dur_std=df.std(), count_kernel=None
    )

    df = df_raw
    df = df.group_by("section_index", "bubble_mode").agg(
        dur=pl.col("dur_bubble_before_kernel").sum()
    )
    df = df.sort("section_index", "bubble_mode")
    df = df.group_by("bubble_mode").agg(
        dur_mean=pl.col("dur").mean(),
        dur_std=pl.col("dur").std(),
    )
    df = df.rename({"bubble_mode": "category"})
    df_bubble = df

    df = df_raw
    df = df.group_by("section_index", "category")
    df = df.agg(dur=pl.col("dur").sum(), count_kernel=pl.len())
    df = df.group_by("category")
    df = df.agg(
        dur_mean=pl.col("dur").mean(),
        dur_std=pl.col("dur").std(),
        count_kernel=pl.col("count_kernel").sum(),
    )
    df = df.sort("dur_mean", descending=True)
    df = pl.concat([pl.DataFrame([row_overall]), df_bubble, df], how="diagonal_relaxed")
    sorted_categories = (
        ["Overall"]
        + list(per_scenario_info.category_info)
        + [
            "Other Kernels",
            "Comp Stream Bubble",
            "Comm Stream Bubble",
            "Launch Overhead",
        ]
    )
    df = df.with_columns(
        category_sort_index=pl.col("category")
        .replace({name: i for i, name in enumerate(sorted_categories)})
        .cast(int)
    ).sort("category_sort_index")
    df = _compute_dur_scaled(df, per_file_info.time_multiplier)
    if "prefill" in pack_name:
        print("Remove launch_overhead category since prefill")
        df = df.filter(pl.col("category") != "Launch Overhead")
    df_agg_by_category = df

    return dict(
        pack_name=pack_name,
        per_file_info=per_file_info,
        per_scenario_info=per_scenario_info,
        df_raw=df_raw,
        df_agg_by_name=df_agg_by_name,
        df_agg_by_category=df_agg_by_category,
    )


def _compute_bubble(df_raw):
    dfs = []
    stream_names = []

    tids = df_raw["tid"].unique().sort()
    assert len(tids) <= 2
    for tid in tids:
        df = df_raw
        df = df.filter(pl.col("tid") == tid)

        has_communication = len(df.filter(pl.col("name").str.contains("dispatch"))) > 0
        stream_name = "Comm Stream" if has_communication else "Comp Stream"

        df = df.sort("ts")
        df = df.with_columns(ts_end=pl.col("ts") + pl.col("dur"))
        df = df.with_columns(prev_ts_end=pl.col("ts_end").shift(1))
        # # `-1` for rounding error
        # df = df.with_columns(curr_start_ge_prev_end=(pl.col('ts') >= pl.col('prev_ts_end') - 1).fill_null(True))
        df = df.with_columns(
            dur_bubble_before_kernel=(pl.col("ts") - pl.col("prev_ts_end")).clip(
                0, 1000000000
            ),
        )
        df = df.with_columns(
            bubble_mode=pl.when(pl.col("dur_bubble_before_kernel") >= 20)
            .then(pl.lit(f"{stream_name} Bubble"))
            .otherwise(pl.lit("Launch Overhead"))
        )
        # display(df)

        dfs.append(df)
        stream_names.append(stream_name)

        # df_violate = df.filter(~pl.col('curr_start_ge_prev_end'))
        # assert len(df_violate) == 0, f"should not have overlap {df_violate=}"

        assert len(set(stream_names)) == len(stream_names)
    return pl.concat(dfs)


PER_FILE_INFOS = {}

PER_FILE_INFOS["deepseek_prefill/primary"] = TracePerFileInfo(
    trace_name="deepseek_profile_prefill.json",
    interest_time_ranges=[(1740462778242549000, 1740462780216747000)],
)
PER_FILE_INFOS["deepseek_decode/primary"] = TracePerFileInfo(
    trace_name="deepseek_profile_decode.json",
    interest_time_ranges=[(1742522672423379000, 1742522672504454000)],
)

PER_FILE_INFOS["sglang_prefill/primary"] = TracePerFileInfo(
    # AC3984
    trace_name="1745667096.7651715-TP-30.trace.json.gz",
    interest_time_ranges=[(2145501388464369, 2145503818742021)],
)
PER_FILE_INFOS["sglang_prefill/fake_perfect_eplb"] = TracePerFileInfo(
    # AC3984
    trace_name="1745649620.7260008-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2128023763399204, 2128025835903362),
        # (2128025957905598, 2128028030168847), # not supprort multi yet
    ],
)
PER_FILE_INFOS["sglang_prefill/bs8192"] = TracePerFileInfo(
    # AC4000
    trace_name="1745764229.7797408-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2242634362146433, 2242635688950158),
        # (2242637163242572, 2242638494643792),
    ],
    time_multiplier=2.0,
)
PER_FILE_INFOS["sglang_prefill_no_tbo/bs8192"] = TracePerFileInfo(
    # AC3999
    trace_name="1745749932.4693627-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2228336827599917, 2228338583260512),
    ],
    time_multiplier=2.0,
)
PER_FILE_INFOS["sglang_prefill_no_tbo/bs8192__fake_perfect_eplb"] = TracePerFileInfo(
    # AC3999
    trace_name="1745982788.1031728-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2461202975243152, 2461204414862208),
    ],
    time_multiplier=2.0,
)

PER_FILE_INFOS["sglang_prefill/sm20__fake_perfect_eplb"] = TracePerFileInfo(
    # AC4046
    trace_name="1745990452.302106-TP-30.trace.json.gz",
    interest_time_ranges=[(2468860275880164, 2468862335909065)],
)
PER_FILE_INFOS["sglang_prefill/sm28__fake_perfect_eplb"] = TracePerFileInfo(
    # AC4046
    trace_name="1745989885.9091156-TP-30.trace.json.gz",
    interest_time_ranges=[(2468306775714218, 2468308842458707)],
)
PER_FILE_INFOS["sglang_prefill/sm32__fake_perfect_eplb"] = TracePerFileInfo(
    # AC4046
    trace_name="1745989129.0789707-TP-30.trace.json.gz",
    interest_time_ranges=[(2467539286165109, 2467541425088855)],
)
PER_FILE_INFOS["sglang_prefill/sm36__fake_perfect_eplb"] = TracePerFileInfo(
    # AC4046
    trace_name="1745990795.4011657-TP-30.trace.json.gz",
    interest_time_ranges=[(2469201317806352, 2469203506584404)],
)

PER_FILE_INFOS["sglang_prefill/sm20"] = TracePerFileInfo(
    # AC4046
    trace_name="1745992176.049016-TP-30.trace.json.gz",
    interest_time_ranges=[(2470584043979478, 2470586639546708)],
)
PER_FILE_INFOS["sglang_prefill/sm28"] = TracePerFileInfo(
    # AC4046
    trace_name="1745991042.73723-TP-30.trace.json.gz",
    interest_time_ranges=[(2469448419631570, 2469450852921364)],
)
PER_FILE_INFOS["sglang_prefill/sm32"] = TracePerFileInfo(
    # AC4046
    trace_name="1745990617.5949163-TP-30.trace.json.gz",
    interest_time_ranges=[(2469027734964434, 2469030185689410)],
)
PER_FILE_INFOS["sglang_prefill/sm36"] = TracePerFileInfo(
    # AC4046
    trace_name="1745991715.3374755-TP-30.trace.json.gz",
    interest_time_ranges=[(2470121645660365, 2470124150769779)],
)

PER_FILE_INFOS["sglang_decode/primary"] = TracePerFileInfo(
    # AC4002
    trace_name="1745921484.9133716-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2399946985814654, 2399947070674961),
    ],
)
PER_FILE_INFOS["sglang_decode/simulated_slow_attn"] = TracePerFileInfo(
    # AC3988
    trace_name="1745924010.6395364-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2402476578882425, 2402476666479454),
    ],
)
PER_FILE_INFOS["sglang_decode/in1800"] = TracePerFileInfo(
    # AC4002
    trace_name="1745731417.4402983-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2209840224855541, 2209840310008215),
    ],
)
PER_FILE_INFOS["sglang_decode_no_tbo/in1800_bs256"] = TracePerFileInfo(
    # AC4001
    trace_name="1745754296.317906-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2232771454731690, 2232771562918629),
    ],
)
PER_FILE_INFOS["sglang_decode_no_tbo/in1800_bs128"] = TracePerFileInfo(
    # AC4001
    trace_name="1745755036.6394377-TP-30.trace.json.gz",
    interest_time_ranges=[
        (2233479608493610, 2233479671290116),
    ],
    time_multiplier=2.0,
)

_HACK_SLOW_ATTN_NUM_REPEAT = 7

PER_SCENARIO_INFOS = {}
PER_SCENARIO_INFOS["deepseek_prefill"] = TracePerScenarioInfo(
    layer_indicator_kernel="void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, at::detail::Array<char*, 3>)",
    layer_indicator_kernel_sparsity=2,
    category_info={
        "Dispatch Barrier": [
            "void dpsk::ep::internode::notify_dispatch<false, 4>(int const*, long*, int, int const*, int*, int const*, int*, int, bool const*, int, int, int, int, int, int, int, int*, int*, int*, int*, void*, void**, int**, int, int, int*, int)",
        ],
        "Dispatch Core": [
            "void dpsk::ep::cuda::get_dispatch_layout<256, 32, 8>(long const*, int*, int*, int*, bool*, int, int, int, int)",
            "void dpsk::ep::internode::dispatch<false, 4, false, 4>(int4*, float*, long*, float*, dpsk::ep::internode::SourceMeta*, int4 const*, float const*, long const*, float const*, int*, int*, int*, int*, int const*, int const*, int const*, int const*, int, int, int, int, int, bool const*, void*, int, int, void**, int, int, int, int, int*)",
        ],
        "Combine Barrier": [
            "void dpsk::ep::internode::cached_notify<false>(int, int, int, int, int*, int, int, int const*, int const*, int*, void*, void**, int**, int, int, int, int*, bool, int)",
        ],
        "Combine Core": [
            "void dpsk::ep::internode::combine<false, 4, __nv_bfloat16, 4, 4, 16, 24>(int4*, float*, bool const*, int4 const*, float const*, int4 const*, int4 const*, int const*, int const*, dpsk::ep::internode::SourceMeta const*, int const*, int const*, int const*, int, int, int, int, void*, int, int, void**, int, int, int, int, int*)"
        ],
        "MoE": [
            "void dpsk::gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)1>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cuda::swiglu_forward_with_weight_and_per_token_cast_to_fp8_with_channels_impl<__nv_bfloat16, false, true, 128>(unsigned char*, __nv_bfloat16 const*, int const*, float const*, int, int, float*, int const*, float, __nv_fp8_interpretation_t)",
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)1>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
        ],
        "Attention Core": [
            "void flash::compute_attn_ws<Flash_fwd_kernel_traits<192, 128, 128, 12, 2, false, 1, cutlass::bfloat16_t, 128>, true, flash::SingleTileScheduler, flash::SeqLenTraits<true> >(flash::CollectiveMainloopFwd<Flash_fwd_kernel_traits<192, 128, 128, 12, 2, false, 1, cutlass::bfloat16_t, 128>, true, flash::SeqLenTraits<true> >::Params, flash::CollectiveEpilogueFwd<Flash_fwd_kernel_traits<192, 128, 128, 12, 2, false, 1, cutlass::bfloat16_t, 128>, flash::SeqLenTraits<true> >::Params, flash::SingleTileScheduler::Params, flash::SeqLenTraits<true>, flash::SeqLenTraits<true>)",
        ],
        "GEMM": [
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 16384u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<24576u, 1536u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 128u, 128u, 5u, 128u, 128u, 2u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<2112u, 7168u, 128u, 128u, 128u, 5u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cutlass::device_kernel<dpsk::grouped_gemm::cuda::fp8_ptp128c_outfmt_head::GemmKernel>(dpsk::grouped_gemm::cuda::fp8_ptp128c_outfmt_head::GemmKernel::Params)",
            "sm90_xmma_gemm_bf16f32_bf16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas",
        ],
    },
)
PER_SCENARIO_INFOS["deepseek_decode"] = TracePerScenarioInfo(
    layer_indicator_kernel="void flash::flash_fwd_splitkv_mla_kernel<Flash_fwd_kernel_traits_mla<576, 64, 64, 8, cutlass::bfloat16_t, 512, true, 0, cutlass::bfloat16_t, cutlass::bfloat16_t>, true, flash::SharedStorageMLA<Flash_fwd_kernel_traits_mla<576, 64, 64, 8, cutlass::bfloat16_t, 512, true, 0, cutlass::bfloat16_t, cutlass::bfloat16_t> > >(Flash_fwd_mla_params)",
    layer_indicator_kernel_sparsity=2,
    category_info={
        "Dispatch": [
            "void dpsk::ep::internode::dispatch_ll<true, 3, 10, 7168>(void*, float*, int*, long*, int*, int*, void*, int*, void*, void const*, long const*, int*, int*, int*, int, int, int, int, int, int, int, int*, int)",
        ],
        "Combine": [
            "void dpsk::ep::internode::combine_ll<3, 10, 7168, 9>(void*, void*, int*, void*, void const*, long const*, float const*, int const*, long const*, int*, int, int*, int, int, int, int, int, int, int, int*, int)",
        ],
        "MoE": [
            "void dpsk::gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 112u, 128u, 6u, 128u, 128u, 1u, (dpsk::gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cuda::batched_swiglu_forward_and_per_token_cast_to_fp8_with_channels_impl<__nv_bfloat16, 128, 512, true, 128>(unsigned char*, __nv_bfloat16 const*, int, int, int, float*, int const*, float, __nv_fp8_interpretation_t)",
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 112u, 128u, 6u, 128u, 128u, 1u, (dpsk::gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
        ],
        "Attention Core": [
            "void flash::flash_fwd_splitkv_mla_kernel<Flash_fwd_kernel_traits_mla<576, 64, 64, 8, cutlass::bfloat16_t, 512, true, 0, cutlass::bfloat16_t, cutlass::bfloat16_t>, true, flash::SharedStorageMLA<Flash_fwd_kernel_traits_mla<576, 64, 64, 8, cutlass::bfloat16_t, 512, true, 0, cutlass::bfloat16_t, cutlass::bfloat16_t> > >(Flash_fwd_mla_params)",
            "void flash::flash_fwd_splitkv_mla_combine_kernel<cutlass::bfloat16_t, float, long, 512, 64>(Flash_fwd_mla_params)",
        ],
        "GEMM": [
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 16384u, 128u, 56u, 128u, 6u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<24576u, 1536u, 128u, 96u, 128u, 6u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 32u, 128u, 8u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_64x3_nn_align8>(cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_64x3_nn_align8::Params)",
            "void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_64x3_tn_align8::Params)",
            "void dpsk::gemm::fp8_gemm_kernel<2112u, 7168u, 128u, 32u, 128u, 8u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void dpsk::gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 56u, 128u, 6u, 128u, 128u, 1u, (dpsk::gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cutlass::Kernel<cutlass_80_tensorop_s16816gemm_bf16_128x64_64x4_tn_align8>(cutlass_80_tensorop_s16816gemm_bf16_128x64_64x4_tn_align8::Params)",
        ],
    },
)

_SGLANG_PREFILL_PARTIAL_INFO = dict(
    # NOTE pick the one at the start of first `combine`
    layer_indicator_kernel="_fwd_kernel_ep_scatter_2",
    category_info={
        "Dispatch Barrier": [
            "void deep_ep::internode::notify_dispatch<false, 4>(int const*, int*, int, int const*, int*, int const*, int*, int, bool const*, int, int, int, int, int, int, int, int*, int*, int*, int*, void*, void**, int**, int, int, int)",
        ],
        "Dispatch Core": [
            "void deep_ep::internode::get_dispatch_layout<256, 32, 8>(long const*, int*, int*, int*, bool*, int, int, int, int)",
            "void deep_ep::internode::dispatch<false, 4, false, 7, 4>(int4*, float*, long*, float*, deep_ep::internode::SourceMeta*, int4 const*, float const*, long const*, float const*, int*, int*, int*, int*, int const*, int const*, int const*, int const*, int, int, int, int, int, bool const*, void*, int, int, void**, int, int, int, int)",
        ],
        "Combine Barrier": [
            "void deep_ep::internode::cached_notify<false>(int, int, int, int, int*, int, int, int const*, int const*, int*, void*, void**, int**, int, int, int, bool, int)",
        ],
        "Combine Core": [
            "void deep_ep::internode::combine<false, 4, __nv_bfloat16, 16, 4, 4, 16, 24>(int4*, float*, bool const*, int4 const*, float const*, int const*, int const*, deep_ep::internode::SourceMeta const*, int const*, int const*, int const*, int, int, int, int, void*, int, int, void**, int, int, int, int)"
        ],
        "MoE": [
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 160u, 128u, 0u, 64u, 9u, 4u, 128u, 128u, 1u, true, (deep_gemm::GemmType)1>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(silu(float const&))>(__nv_bfloat16*, __nv_bfloat16 const*, int)",
            "void per_token_group_quant_8bit_kernel<__nv_bfloat16, c10::Float8_e4m3fn, false>(__nv_bfloat16 const*, void*, float*, int, int, int, float, float, float, int, int)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 160u, 128u, 0u, 64u, 9u, 4u, 128u, 128u, 1u, true, (deep_gemm::GemmType)1>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
        ],
        "Attention Core": [
            "flash::prepare_varlen_num_blocks_kernel(int, int, int, int const*, int const*, int const*, int const*, int const*, int const*, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int*, int*, bool)",
            "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<192> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, false, false, false, true, true, false, false, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, false, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 256, 32, false, false, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<128>, cute::C<192> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, false, false, false, true, true, false, false, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, false, false, false>, flash::VarlenDynamicPersistentTileScheduler<128, 256, 32, false, false, true> > >::Params)",
        ],
        "GEMM": [
            "void deep_gemm::fp8_gemm_kernel<7168u, 16384u, 256u, 128u, 128u, 0u, 128u, 1u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<24576u, 1536u, 256u, 128u, 128u, 0u, 128u, 1u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 256u, 128u, 128u, 0u, 128u, 1u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<32768u, 512u, 256u, 128u, 128u, 0u, 128u, 1u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 256u, 128u, 128u, 0u, 128u, 1u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<2112u, 7168u, 256u, 112u, 128u, 0u, 32u, 1u, 3u, 128u, 128u, 2u, false, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            # ============ for bs8192 ============
            "void deep_gemm::fp8_gemm_kernel<7168u, 16384u, 256u, 120u, 128u, 0u, 0u, 1u, 3u, 128u, 128u, 2u, false, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 256u, 120u, 128u, 0u, 0u, 1u, 3u, 128u, 128u, 2u, false, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
        ],
    },
)
PER_SCENARIO_INFOS["sglang_prefill"] = TracePerScenarioInfo(
    layer_indicator_kernel_sparsity=2,
    **_SGLANG_PREFILL_PARTIAL_INFO,
)
PER_SCENARIO_INFOS["sglang_prefill_no_tbo"] = TracePerScenarioInfo(
    layer_indicator_kernel_sparsity=1,  # NOTE
    **_SGLANG_PREFILL_PARTIAL_INFO,
)

_SGLANG_DECODE_PARTIAL_INFO = dict(
    layer_indicator_kernel="void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<64>, cute::C<64> >, 512, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, true, false, false, true, true, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<512>, cute::C<64> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 256, 128, true, true, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<64>, cute::C<64> >, 512, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, true, false, false, true, true, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<512>, cute::C<64> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 256, 128, true, true, true> > >::Params)",
    kernels_to_add_index=[
        "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)",
    ],
    category_info={
        "Dispatch": [
            "void deep_ep::internode_ll::dispatch<true, 3, 10, 7168>(void*, float*, int*, long*, int*, void*, int*, void*, void const*, long const*, int*, int*, int*, int, int, int, int, int, int, int, int)",
        ],
        "Combine": [
            "void deep_ep::internode_ll::combine<3, 10, 7168, 9>(void*, void*, int*, void*, void const*, long const*, float const*, int const*, long const*, int*, int, int*, int, int, int, int, int, int, int, int, bool)",
        ],
        "MoE": [
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 256u, 128u, 128u, 0u, 128u, 4u, 3u, 128u, 128u, 1u, true, (deep_gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "_silu_and_mul_post_quant_kernel",
            # there is an elementwise kernel, not put here?
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 256u, 112u, 128u, 0u, 32u, 4u, 3u, 128u, 128u, 1u, true, (deep_gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            # ============ for no_tbo bs256 ============
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 256u, 128u, 128u, 0u, 128u, 4u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 256u, 112u, 128u, 0u, 32u, 4u, 3u, 128u, 128u, 2u, true, (deep_gemm::GemmType)2>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
        ],
        "Attention Core": [
            "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)",
            "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 3, 64, 64>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 64, 64>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
            "void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<2> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<2> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<2> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<2> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&)::{lambda(int)#1})",
            "flash::prepare_varlen_num_blocks_kernel(int, int, int, int const*, int const*, int const*, int const*, int const*, int const*, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int*, int*, bool)",
            "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<64>, cute::C<64> >, 512, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, true, false, false, true, true, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<512>, cute::C<64> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 256, 128, true, true, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<64>, cute::C<64> >, 512, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, true, false, false, true, true, false>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<512>, cute::C<64> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 256, 128, true, true, true> > >::Params)",
            "void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 6, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> >(flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 6, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90>::Params)",
            *[
                f"{i}#void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)"
                for i in range(20)
                if i % (_HACK_SLOW_ATTN_NUM_REPEAT + 1) != 0
            ],
        ],
        "GEMM": [
            "void deep_gemm::fp8_gemm_kernel<7168u, 16384u, 64u, 112u, 128u, 0u, 32u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<24576u, 1536u, 128u, 96u, 128u, 0u, 64u, 1u, 4u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 64u, 64u, 128u, 0u, 128u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            # NOTE
            "0#void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)",
            f"{_HACK_SLOW_ATTN_NUM_REPEAT + 1}#void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params)",
            "void deep_gemm::fp8_gemm_kernel<2112u, 7168u, 64u, 32u, 128u, 0u, 64u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 64u, 112u, 128u, 0u, 32u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x64_64x4_tn_align8>(cutlass_80_tensorop_s16816gemm_bf16_128x64_64x4_tn_align8::Params)",
            # ============ for no_tbo bs256 ============
            "void deep_gemm::fp8_gemm_kernel<7168u, 16384u, 128u, 112u, 128u, 0u, 32u, 1u, 6u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<24576u, 1536u, 256u, 96u, 128u, 0u, 64u, 1u, 3u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)"
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 64u, 128u, 128u, 0u, 128u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<2112u, 7168u, 64u, 64u, 128u, 0u, 128u, 1u, 8u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void deep_gemm::fp8_gemm_kernel<7168u, 2048u, 128u, 112u, 128u, 0u, 32u, 1u, 6u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8::Params)",
            "void deep_gemm::fp8_gemm_kernel<24576u, 1536u, 256u, 96u, 128u, 0u, 64u, 1u, 3u, 128u, 128u, 1u, true, (deep_gemm::GemmType)0>(__nv_bfloat16*, float*, int*, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)",
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
        ],
    },
)
PER_SCENARIO_INFOS["sglang_decode"] = TracePerScenarioInfo(
    layer_indicator_kernel_sparsity=2,
    **_SGLANG_DECODE_PARTIAL_INFO,
)
PER_SCENARIO_INFOS["sglang_decode_no_tbo"] = TracePerScenarioInfo(
    layer_indicator_kernel_sparsity=1,  # NOTE
    **_SGLANG_DECODE_PARTIAL_INFO,
)


def compute_manual_df():
    rows = []

    pack_name = "manual_prefill/unit_test"
    rows += [
        dict(
            category="Dispatch Core",
            dur_mean=3630 * 2,
            category_sort_index=1,
            pack_name=pack_name,
        ),
        dict(
            category="Combine Core",
            dur_mean=7120 * 2,
            category_sort_index=2,
            pack_name=pack_name,
        ),
        dict(
            category="MoE",
            # activation time: 300/6100, so we add it here
            dur_mean=(5254 + 300) * 2,
            category_sort_index=3,
            pack_name=pack_name,
        ),
    ]

    pack_name = "manual_decode/unit_test"
    rows += [
        dict(
            category="Dispatch",
            dur_mean=30 * 2,
            category_sort_index=1,
            pack_name=pack_name,
        ),
        dict(
            category="Combine",
            dur_mean=39 * 2,
            category_sort_index=2,
            pack_name=pack_name,
        ),
        dict(
            category="MoE",
            # activation time: 8/126, so we add it here
            dur_mean=(88.66 + 8) * 2,
            category_sort_index=3,
            pack_name=pack_name,
        ),
    ]

    df = pl.DataFrame(rows)
    df = df.with_columns(dur_mean_scaled=pl.col("dur_mean"))
    return df


def _compute_dur_scaled(df, time_multiplier: float):
    return df.with_columns(
        dur_mean_scaled=pl.col("dur_mean") * time_multiplier,
        dur_std_scaled=pl.col("dur_std") * time_multiplier,
    )
