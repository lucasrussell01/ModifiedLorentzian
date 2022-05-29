import numpy as np
from scipy.special import gamma
from scipy.special import wofz

i = complex(0, 1)


def value_to_index(inarr, target):
    """
    Given an input array, inarr, and a target value return the index of the
    input array closest to the target

    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-
    arrayand
    http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-
    return-the-index-of-array-in-python
    def getnearpos(array,value):
        idx = (np.abs(array-value)).argmin() ## or idx = numpy.argmin
        (numpy.abs(A - target))
        return idx

    :param inarr: 1-dimensional np-array
    :param target: value request to find closest member of inarr
    :return: index of inarr with value closest to target
   """
    assert isinstance(inarr, np.ndarray), "Input array must be a numpy.ndarray"
    assert inarr.ndim == 1, "Input array must be one-dimensional"
    idx = (np.abs(inarr-target)).argmin()
    return idx


class analytical:
    """This class contains methods to evaluate the analytical convolution of a
    modified Lorentzian convolved with a Gaussian, and its partial
    derivatives."""

    def __init__(self, variable, nu, l_0, w_g, w_l, phi=1):
        "This method initialises all of the variables required"

        self.c_g = 2*np.sqrt(np.log(2))
        self.variable = variable
        self.nu = nu
        self.l_0 = l_0
        self.w_g = w_g
        self.w_l = w_l
        self.phi = phi

    def evaluate(self):
        """This method evaluates the convolution integral anaytically using a
        a second order Taylor series aproximation."""

        
        w = self.w_l/2
        c_nu = (self.nu*w**(self.nu-1))/(2*gamma(1/self.nu)*gamma(1-1/self.nu))
        c = (c_nu/np.sqrt(np.pi))*(self.c_g/self.w_g)**self.nu
        alpha = (self.c_g * np.abs(self.variable-self.l_0))/self.w_g
        al = (self.c_g * (self.variable-self.l_0))/self.w_g
        beta = (self.w_l*self.c_g)/(2*self.w_g)
        c_2 = 2/(self.nu*(self.nu-1)*alpha**(self.nu-2))
        A = (self.nu*alpha*np.sign(al))/(self.nu*(self.nu-1))
        B = np.sqrt((2*(alpha**self.nu+beta**self.nu))/(self.nu*(self.nu-1) * alpha**(self.nu-2))-A**2)
        F = wofz(A + i*B)
        C_f = c*c_2*np.pi/B
        Re_F = np.real(F)
        # Find peak for singularity
        peak = value_to_index(alpha, 0)
        solution = self.phi*C_f*Re_F
        solution[peak] = self.phi*c*np.sqrt(np.pi)/(beta**self.nu)
        #return (solution)
        return solution
        #return A, B, solution

    def partialderivative(self, parameter):
        """This method evaluates the partial derivative of the analytical fit
        with respect to a parameter "l_0", "w_g", "w_l" or "phi". For
        singularities at (l-l0)=0 we set the partial derivative to the value
        taken by the corresponding derivative of the Voigt function """

        # Definitions of variables and constants
        w = self.w_l/2
        c_nu = (self.nu*w**(self.nu-1))/(2*gamma(1/self.nu)*gamma(1-1/self.nu))
        c = (c_nu/np.sqrt(np.pi))*(self.c_g/self.w_g)**self.nu
        alpha = (self.c_g * np.abs(self.variable-self.l_0))/self.w_g
        al = (self.c_g * (self.variable-self.l_0))/self.w_g
        beta = (self.w_l*self.c_g)/(2*self.w_g)
        c_2 = 2/(self.nu*(self.nu-1)*alpha**(self.nu-2))
        A = (alpha*np.sign(al))/((self.nu-1))
        B = np.sqrt((2*(alpha**self.nu+beta**self.nu))/(self.nu*(self.nu-1) * alpha**(self.nu-2))-A**2)
        C_f = c*c_2*np.pi/B
        pos = np.abs(self.variable - self.l_0)
        # PDs of K wrt to A and B
        z = A + i*B
        K = np.real(wofz(z))
        L = np.imag(wofz(z))
        dKdA = B*L - A*K
        dKdB = A*L + B*K - 1/(np.sqrt(np.pi))
        # Find peak for singularity
        peak = value_to_index(alpha, 0)

        if (parameter == "l_0"):  # derivative wrt l_0
            # derivative of A wrt l_0
            dAdl0 = -self.c_g/((self.nu-1)*self.w_g)
            # derivative of B wrt l_0
            nu=self.nu
            c_g = self.c_g
            w_g = self.w_g
            CB1 = 2/(nu*(nu-1))
            CB2 = -CB1*np.sign(al)*c_g/(w_g*alpha**(2*nu-4))
            dB1dl0 = CB2*(2*alpha**(2*nu-3)-beta**nu *(nu-2)*alpha**(nu-3))
            dA2dl0 = -2*A*c_g/(w_g*(nu-1))
            dBdl0 = (dB1dl0 - dA2dl0)/(2*B) 
            # derivative of C wrt l_0
            CC1 = (2*np.pi*c)/(nu*(nu-1))
            dCdl0 = -CC1*(1/B**2 * dBdl0*alpha**(2-nu)+1/B * (2-nu) *alpha**(1-nu)*np.sign(al)*c_g/w_g)
            # Combination of all elements
            dIdl0 = self.phi*dCdl0*K+C_f*self.phi*(dKdA*dAdl0+dKdB*dBdl0)
            dIdl0[peak] = 0  # return 0 for peak as each side tends to +/- infty
            return dIdl0
        elif parameter == "w_g":  # derivative wrt w_g
            # derivative of A wrt w_g
            dAdwg = -al/((self.nu-1)*self.w_g)
            # derivative of B wrt w_g
            CB1 = 2/(self.nu*(self.nu-1))
            CB2 = -CB1/(self.w_g*alpha**(2*self.nu-4))
            dB1dwg = CB2*(self.nu*alpha**(self.nu-2)*(alpha**(self.nu-1)*np.sign(al)*al + beta**self.nu)
                            -(alpha**self.nu+beta**self.nu)*(self.nu-2)*alpha**(self.nu-3)*np.sign(al)*al)
            dA2dwg = 2*A*dAdwg
            dBdwg = (dB1dwg - dA2dwg)/(2*B) 
            # derivative of C wrt w_g
            CC1 = 2*np.pi/(self.nu*(self.nu-1))
            CC2 = CC1*c/(B**2*alpha**(2*self.nu-4))
            dCdwg = CC2*(-self.nu*B*alpha**(self.nu-2)/self.w_g - dBdwg*alpha**(self.nu-2) +
                            B*(self.nu-2)*alpha**(self.nu-3)*np.sign(al)*al/self.w_g)
            # Combination of all elements
            dIdwg = self.phi*dCdwg*K+C_f*self.phi*(dKdA*dAdwg + dKdB*dBdwg)
            dIdwg[peak] = dIdwg[peak-1]+(dIdwg[peak-1]-dIdwg[peak-2])  # return previous value + previous gradient for peak
            return(dIdwg)
        elif parameter == "w_l":  # derivative wrt w_l
            # derivative of B wrt w_l
            dBdwl = self.c_g*beta**(self.nu-1)/(2*(self.nu-1)*self.w_g*B*alpha**(self.nu-2))
            # derivative of C wrt w_l
            CC1 = 2*np.pi/(self.nu*(self.nu-1)*alpha**(self.nu-2))
            dCdwl = CC1/B**2 * ((self.nu-1)*c/self.w_l*B - dBdwl*c)
            # Combination of all elements
            dIdwl = self.phi*dCdwl*K+C_f*self.phi*(dKdB*dBdwl)
            dIdwl[peak] = dIdwl[peak-1]+(dIdwl[peak-1]-dIdwl[peak-2])  # return previous value + previous gradient for peak
            return dIdwl
        elif parameter == "phi":  # derivative wrt area
            dIdphi = C_f*K
            return dIdphi
        else:
            raise Exception("Please choose 'l_0', 'w_g', 'w_l' or 'phi'")
