df_pack_arr = []
for name in [
    'deepseek_prefill/primary',
    'sglang_prefill/primary',
    'sglang_prefill/fake_perfect_eplb',
    'sglang_prefill/bs8192',
    'sglang_prefill_no_tbo/bs8192',
    'sglang_prefill_no_tbo/bs8192__fake_perfect_eplb',

    'deepseek_decode/primary',
    'sglang_decode/primary',
    'sglang_decode/simulated_slow_attn',
    'sglang_decode/in1800',
    'sglang_decode_no_tbo/in1800_bs256',
    'sglang_decode_no_tbo/in1800_bs128',

    # not used yet
    # 'sglang_prefill/sm20',
    # 'sglang_prefill/sm28',
    # 'sglang_prefill/sm32',
    # 'sglang_prefill/sm36',
    # 'sglang_prefill/sm20__fake_perfect_eplb',
    # 'sglang_prefill/sm28__fake_perfect_eplb',
    # 'sglang_prefill/sm32__fake_perfect_eplb',
    # 'sglang_prefill/sm36__fake_perfect_eplb',
]:
    df_pack = trace_analyzer.analyze_trace(name)
    display(df_pack['df_agg_by_category'])
    df_pack_arr.append(df_pack)

df_pack = df_pack_arr[-1]
with pl.Config(fmt_str_lengths=1000, tbl_cols=-1, tbl_rows=-1):
    display(df_pack['df_agg_by_category'])
    if 1:
        display(df_pack['df_agg_by_name'].filter(pl.col('category') == 'Other Kernels'))
        display(df_pack['df_agg_by_name'])
if 1:
    display(df_pack['df_raw'])

# ========================================================================

df_agg_by_category = pl.concat(
    [x['df_agg_by_category'].with_columns(pack_name=pl.lit(x['pack_name'])) for x in df_pack_arr] +
    [trace_analyzer.compute_manual_df()],
    how='diagonal_relaxed',
    )
df_agg_by_category

# ========================================================================

for stage in ['prefill', 'decode']:
    df = df_agg_by_category.filter(pl.col('pack_name').str.contains(stage)).drop('count_kernel')
    fig = px.bar(
        df.to_pandas(),
        y='category', x='dur_mean_scaled', error_x='dur_std_scaled',
        color='pack_name', barmode='group', orientation='h',
    )
    fig.update_layout(height=700)
    fig.update_layout(yaxis={'autorange': 'reversed'})
    fig.show()

# ========================================================================

for title, output_name, interest_legend_and_pack_names in [
    (
            'Prefill Time Breakdown',
            'profile-prefill.png',
            [
                ('Unit Test', 'manual_prefill/unit_test',),
                ('DeepSeek Profile', 'deepseek_prefill/primary',),
                ('SGLang', 'sglang_prefill/primary',),
                ('SGLang + Fake Perfect EPLB', 'sglang_prefill/fake_perfect_eplb',),
            ],
    ),
    (
            'Decode Time Breakdown',
            'profile-decode.png',
            [
                ('Unit Test', 'manual_decode/unit_test',),
                ('DeepSeek Profile', 'deepseek_decode/primary',),
                ('SGLang', 'sglang_decode/primary',),
                ('SGLang + Simulated Slow Attention', 'sglang_decode/simulated_slow_attn',),
            ],
    ),
    (
            'TBO Ablation (Prefill)',
            'tbo-breakdown-prefill.png',
            [
                ('TBO + 16k tok/device', 'sglang_prefill/primary',),
                ('TBO + 8k tok/device', 'sglang_prefill/bs8192',),
                ('No TBO + 8k tok/device', 'sglang_prefill_no_tbo/bs8192',),
            ],
    ),
    (
            'TBO Ablation (Decode)',
            'tbo-breakdown-decode.png',
            [
                ('TBO + 256 tok/device', 'sglang_decode/in1800',),
                ('No TBO + 256 tok/device', 'sglang_decode_no_tbo/in1800_bs256',),
                ('No TBO + 128 tok/device', 'sglang_decode_no_tbo/in1800_bs128',),
            ],
    ),

    # [
    #     'sglang_prefill/sm20',
    #     'sglang_prefill/primary',
    #     'sglang_prefill/sm28',
    #     'sglang_prefill/sm32',
    #     'sglang_prefill/sm36',
    # ],
    # [
    #     'sglang_prefill/sm20__fake_perfect_eplb',
    #     'sglang_prefill/fake_perfect_eplb',
    #     'sglang_prefill/sm28__fake_perfect_eplb',
    #     'sglang_prefill/sm32__fake_perfect_eplb',
    #     'sglang_prefill/sm36__fake_perfect_eplb',
    # ],
]:
    print(interest_legend_and_pack_names)

    interest_names = [x[1] for x in interest_legend_and_pack_names]

    df = df_agg_by_category.filter(pl.col('pack_name').is_in(interest_names)).drop('count_kernel')
    df = df.with_columns(sort_index=pl.col('pack_name').replace({name: i for i, name in enumerate(interest_names)})).sort('sort_index')
    # display(df)

    if 1:
        fig = px.bar(
            df.to_pandas(),
            y='category', x='dur_mean_scaled', error_x='dur_std_scaled',
            color='pack_name', barmode='group', orientation='h',
        )
        fig.update_layout(height=600)
        fig.show()

    #########################

    df = df.with_columns(
        dur_mean_scaled=pl.col('dur_mean_scaled') / 1000,
        dur_std_scaled=pl.col('dur_std_scaled') / 1000,
    )
    df = df.to_pandas()

    if 'prefill' in interest_legend_and_pack_names[0][1]:
        categories = [
            'Overall',
            'Dispatch Barrier',
            'Dispatch Core',
            'Combine Barrier',
            'Combine Core',
            'MoE',
            'Attention Core',
            'GEMM',
            'Other Kernels',
            'Comp Stream Bubble',
            'Comm Stream Bubble',
        ]
    elif 'decode' in interest_legend_and_pack_names[0][1]:
        categories = [
            'Overall',
            'Dispatch',
            'Combine',
            'MoE',
            'Attention Core',
            'GEMM',
            'Other Kernels',
            'Launch Overhead',
        ]
    else:
        raise

    categories = categories[::-1]
    # categories = df['category'].unique()[::-1]

    pack_names = df['pack_name'].unique()[::-1]
    print(pack_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_packs = len(pack_names)
    bar_height = 0.8 / n_packs

    for i, pack in enumerate(pack_names):
        pack_data = df[df['pack_name'] == pack]

        y_positions = np.arange(len(categories))
        y_offset = (i - n_packs/2 + 0.5) * bar_height

        category_data = {row['category']: (row['dur_mean_scaled'], row['dur_std_scaled'])
                         for _, row in pack_data.iterrows()}

        values = []
        errors = []
        for cat in categories:
            if cat in category_data:
                values.append(category_data[cat][0])
                errors.append(category_data[cat][1])
            else:
                values.append(0)
                errors.append(0)

        label = {pack_name: text for text, pack_name in interest_legend_and_pack_names}[pack]

        ax.barh(
            y_positions + y_offset, values, height=bar_height,
            xerr=errors, label=label, capsize=3,
            color=google_colors[['blue', 'yellow', 'green', 'red'][i]],
            zorder=100,
            error_kw=dict(ecolor='gray', lw=0.5, capsize=2, capthick=0.5),
            )

    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Scaled Mean Time (ms) per Layer')
    ax.set_ylabel('Category')
    ax.set_xlim(0, None)
    ax.legend(
        reverse=True,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.grid(axis='x', color='#c5c5c5', linestyle='-', linewidth=0.5, alpha=0.5, zorder=-100)

    plt.title(title + '\n\n')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.7)

    d = Path('/Volumes/MyExternal/ExternalRefCode/lm-sys.github.io-private/public/images/blog/large_scale_ep')
    plt.savefig(str(d / output_name))
    plt.show()

# ========================================================================

# ========================================================================
