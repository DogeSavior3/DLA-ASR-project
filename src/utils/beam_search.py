import numpy as np


def ctc_beam_search(log_probs, beam_size, ind2char):
    blank = "^"
    beam = {"": (0.0, -np.inf)}

    for T in log_probs:
        next_beam = {}

        for pref, (p_blank, p_not_blank) in beam.items():
            for idx, log_p in enumerate(T):
                sym = ind2char[idx]

                if sym == blank:
                    if pref not in next_beam:
                        next_beam[pref] = (-np.inf, -np.inf)

                    prev_p_blank, prev_p_not_blank = next_beam[pref]
                    next_beam[pref] = (
                        np.logaddexp(
                            prev_p_blank, np.logaddexp(p_blank, p_not_blank) + log_p
                        ),
                        prev_p_not_blank,
                    )

                elif pref and sym == pref[-1]:
                    if pref not in next_beam:
                        next_beam[pref] = (-np.inf, -np.inf)

                    prev_p_blank, prev_p_not_blank = next_beam[pref]
                    next_beam[pref] = (
                        prev_p_blank,
                        np.logaddexp(prev_p_not_blank, p_not_blank + log_p),
                    )

                    new_pref = pref + sym
                    if new_pref not in next_beam:
                        next_beam[new_pref] = (-np.inf, -np.inf)

                    prev_p_blank, prev_p_not_blank = next_beam[new_pref]
                    next_beam[new_pref] = (
                        prev_p_blank,
                        np.logaddexp(prev_p_not_blank, p_blank + log_p),
                    )

                else:
                    new_pref = pref + sym
                    if new_pref not in next_beam:
                        next_beam[new_pref] = (-np.inf, -np.inf)

                    prev_p_blank, prev_p_not_blank = next_beam[new_pref]
                    next_beam[new_pref] = (
                        prev_p_blank,
                        np.logaddexp(
                            prev_p_not_blank, np.logaddexp(p_blank, p_not_blank) + log_p
                        ),
                    )

        beam = {
            k: v
            for _, k, v in sorted(
                [(np.logaddexp(*v), k, v) for k, v in next_beam.items()], reverse=True
            )[:beam_size]
        }

    return [
        (k, score)
        for score, k in sorted((np.logaddexp(*v), k) for k, v in beam.items())[::-1]
    ]
