import numpy as np

def bias_Tinker10(m_range, m_fn):
    '''
    Dark matter bias function derived in Tinker (2010)

        http://dx.doi.org/10.1088/0004-637X/724/2/878

    '''
    nu = np.sqrt(m_fn.nu)
    y = np.log10(m_fn.delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

    return(1 - A * nu ** a / (nu ** a + m_fn.delta_c ** a) +
           B * nu ** b + C * nu ** c)

