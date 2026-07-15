evaluate(env, network, recurrent_fn, key, n_batch, n_step, num_sim, discount):

    _eval_rollout (jitté, calqué sur _rollout mais SANS auto_reset):

        init:
            keys  = split(key, n_batch)
            state = vmap(env.init)(keys)
            step  = vmap(env.step)          # env NU (pas d'auto_reset)

        body_game(carry=(state, key), _):
            key, key_mcts, key_step = split(key, 3)

            mcts_out = search(network, recurrent_fn, state, key_mcts, num_sim)

            action = argmax(mcts_out.action_weights, axis=-1)   # GREEDY (vs sample)

            state = step(state, action, split(key_step, n_batch))
            # slot terminé -> pgx GÈLE l'état : reward=0, terminated reste True

            collect: (reward = state.rewards[:,0], terminated = state.terminated)
            return (state, key), (reward, terminated)
        
        scan body_game sur n_step
            -> reward     [n_step, n_batch]
            -> terminated [n_step, n_batch]

        # --- métriques post-scan ---

        done   = terminated.any(axis=0)                # [n_batch] : slot gagné dans l'horizon ?
        # (le gel pgx garantit qu'un seul terminated=True par slot gagnant)

        # retour actualisé z = Σ γ^t r_t  (backward scan, PAS de cut : 1 partie/slot)
        # ou forward : discounts = γ ** arange(n_step); ret = (reward * discounts[:,None]).sum(0)
        ret = discounted_return(reward, discount)      # [n_batch]

        # longueur = 1er step où terminated (argmax), valide seulement si done
        length = argmax(terminated, axis=0) + 1        # [n_batch]

        # --- agrégation (masquée par done) ---
        success_rate = done.mean()
        mean_return  = ret.mean()                       # échecs contribuent 0 -> OK
        mean_length  = (length * done).sum() / done.sum()   # moyenne sur gagnants seulement
        # garde-fou div/0 si aucun slot gagnant : where(done.sum()>0, ..., nan/0)

        return {success_rate, mean_return, mean_length}

    metrics = _eval_rollout(...)
    return device_get(metrics)
