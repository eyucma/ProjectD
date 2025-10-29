import numpy as np

def phi(x:float|np.array)->float|np.array:
    """
    Approximation of the standard normal cdf for fast computation
    """

    if x==0:
        return 0.5
    elif x>0:
        t=1/(1+x/np.sqrt(8))
        return 1-t/2*np.exp(-x**2/2+sum(a(i)*t**i for i in range(10)))
    else:
        return 1-phi(-x)

def BlackScholes(S:float|np.array,K:float|np.array,T:float|np.array,r:float|np.array,q:float|np.array,vol:float|np.array,call=True, approx=True):
    """
    Compute BlackScholes options price.

    Parameters:
        price: observed option price
        S: underlying spot price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        q: dividend yield
        vol: volatility
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    dp=(np.log(s/c)+(r+sig**2/2)*t)/sig/np.sqrt(t)
    dm=(np.log(s/c)+(r-sig**2/2)*t)/sig/np.sqrt(t)
    return s*phi(dp)-c*np.exp(-r*t)*phi(dm)