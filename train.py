import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_lr(step,d_model=256,warmup=4000):
    step=max(step,1)
    t1=step**-0.5
    t2=step*(warmup**-1.5)
    return (d_model**-0.5)*min(t1,t2)
