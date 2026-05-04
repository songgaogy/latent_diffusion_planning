#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

resolve_ldp_python() {
    if [[ -n "${PY:-}" ]]; then
        if [[ -x "$PY" ]]; then
            printf '%s\n' "$PY"
            return 0
        fi
        printf 'PY is set but not executable: %s\n' "$PY" >&2
        return 1
    fi

    local candidate
    for candidate in \
        "$HOME/anaconda3/envs/ldp/bin/python" \
        "$HOME/miniconda3/envs/ldp/bin/python" \
        "/home/anaconda3/envs/ldp/bin/python"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        candidate="$conda_base/envs/ldp/bin/python"
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    printf 'Could not find ldp python. Set PY=/path/to/envs/ldp/bin/python.\n' >&2
    return 1
}
