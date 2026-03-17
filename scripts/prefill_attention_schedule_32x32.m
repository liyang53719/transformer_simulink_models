function cfg = prefill_attention_schedule_32x32()
%PREFILL_ATTENTION_SCHEDULE_32X32 Default micro-architecture schedule for a 32x32 prefill array.

    cfg = struct();
    cfg.array_rows = 32;
    cfg.array_cols = 32;
    cfg.tile_seq = 64;
    cfg.tile_k = 256;
    cfg.tile_out = 256;
    cfg.x_bank_count = 32;
    cfg.psum_bank_count = 32;
    cfg.kv_bank_count = 16;
    cfg.q_heads_per_kv = 6;
    cfg.kv_phase_first = 1;
    cfg.score_scale = 1;
    cfg.online_softmax_en = 1;
    cfg.scorev_enable = 1;
    cfg.phase_order = {'kv_cache_fill', 'q_head_stream', 'score_softmax', 'value_reduce'};
end