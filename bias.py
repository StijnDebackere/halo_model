import numpy as np

def bias_Tinker10(m_range, nu, delta_c=1.686, delta_h=200.):
    '''
    Dark matter bias function derived in Tinker (2010)

        http://dx.doi.org/10.1088/0004-637X/724/2/878

    '''
    y = np.log10(delta_h)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

    return(1 - A * nu ** a / (nu ** a + delta_c ** a) +
           B * nu ** b + C * nu ** c)

def bias_SMT(m_range, nu, delta_c=1.686):
    a = 0.707
    sa = np.sqrt(a)
    b = 0.5
    c = 0.6

    return (1 + 1./(sa * delta_c) * (sa * a * nu**2 + sa * b * (a*nu**2)**(1-c)
                                     -(a*nu**2)**c / ((a*nu**2)**c +
                                                      b*(1-c)*(1-c/2.))))

