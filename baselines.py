import numpy as np

def naive_policy(obs, params):
    #Order up to target level T each day
    inv, forecast, dow = obs
    T = params.get("target", 50)
    order = max(int(T - inv), 0)
    return order

def sS_policy(obs, params):
    #(s, S) policy: if inv < s, order up to S, otherwise order 0
    inv, forecast, dow = obs
    s, S = params.get("s", 20), params.get("S", 80)
    return max(int(S - inv), 0) if inv < s else 0



    