"""
Data Envelopment Analysis implementation

Sources:
Sherman & Zhu (2006) Service Productivity Management, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
http://deazone.com/en/resources/tutorial

"""

import numpy as np
from scipy.optimize import fmin_slsqp


class DEA(object):

    def __init__(self, inputs, outputs):
        """
        Initialize the DEA object with input data
        n = number of entities (observations)
        m = number of inputs (variables, features)
        r = number of outputs
        :param inputs: inputs, n x m numpy array
        :param outputs: outputs, n x r numpy array
        :return: self
        """

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float64)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float64)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float64)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas

        # names
        self.names = []

    def __efficiency(self, unit):
        """
        Efficiency function with already computed weights
        :param unit: which unit to compute for
        :return: efficiency
        """

        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        return (numerator/denominator)[unit]

    def __target(self, x, unit):
        """
        Theta target function for one unit
        :param x: combined weights
        :param unit: which production unit to compute
        :return: theta
        """
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator/denominator

    def __constraints(self, x, unit):
        """
        Constraints for optimization for one unit
        :param x: combined weights
        :param unit: which production unit to compute
        :return: array of constraints
        """

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):
        """
        Optimization of the DEA model
        Use: http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.linprog.html
        A = coefficients in the constraints
        b = rhs of constraints
        c = coefficients of the target function
        :return:
        """
        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)

    def name_units(self, names):
        """
        Provide names for units for presentation purposes
        :param names: a list of names, equal in length to the number of units
        :return: nothing
        """

        assert(self.n == len(names))

        self.names = names

    def fit(self):
        """
        Optimize the dataset, generate basic table
        :return: table
        """

        self.__optimize()  # optimize

        print("Final thetas for each unit:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.efficiency):
            if len(self.names) > 0:
                name = "Unit %s" % self.names[n]
            else:
                name = "Unit %d" % (n+1)
            print("%s theta: %.4f" % (name, eff))
            print("\n")
        print("---------------------------\n")


if __name__ == "__main__":
    X = np.array([
       [ 21.1 ,   0.9 ,  13.6 ],
       [ 26.8 ,   0.7 ,  22.5 ],
       [ 65.6 ,   1.3 ,  53.9 ],
       [ 19.4 ,   0.8 ,  14.1 ],
       [ 30.4 ,   0.9 ,  18.7 ],
       [ 61.6 ,   1.1 ,  28.1 ],
       [ 21.8 ,   1.2 ,  22.5 ],
       [ 20.9 ,   0.7 ,  14.4 ],
       [ 29.4 ,   0.8 ,  25.6 ],
       [164.5 ,   1.6 ,  55.9 ],
       [ 15.5 ,   0.7 ,  16.5 ],
       [ 20.6 ,   0.6 ,  22.  ],
       [ 16.4 ,   1.  ,  20.2 ],
       [ 18.8 ,   0.6 ,  20.6 ],
       [ 17.56,   0.75,  18.  ],
       [ 16.16,   0.65,  26.3 ],
       [ 20.3 ,   0.7 ,  16.  ],
       [ 15.97,   0.7 ,  15.7 ]
       ])
    y = np.array([
       [  3.  ,   9.7 ,  27.9 ,  19.2 ,  15.7 ],
       [  5.  , 104.1 ,  39.8 ,  23.5 ,  19.7 ],
       [  1.7 , 143.7 , 174.9 , 144.3 ,  76.9 ],
       [  3.7 ,  -2.  ,  17.  ,  16.3 ,  17.1 ],
       [  3.  , 133.4 ,  77.9 ,  59.5 ,  41.8 ],
       [  2.  , 149.6 ,  47.4 ,  37.1 , 108.3 ],
       [ 14.  ,  12.7 ,  17.  ,  22.  ,  20.2 ],
       [  2.9 ,  34.1 ,  20.4 ,  13.  ,  16.7 ],
       [  1.4 ,  52.9 ,  58.2 ,  52.6 , 176.  ],
       [  0.5 , 162.  ,  72.5 ,  96.2 , 148.7 ],
       [  1.7 ,  40.8 ,  16.9 ,  11.2 ,  17.6 ],
       [  1.  , -26.  ,   5.  ,   9.3 ,   8.7 ],
       [ 13.  ,   8.1 ,  13.1 ,  11.  ,  16.2 ],
       [  1.7 ,  44.9 ,   8.7 ,   3.2 ,  13.2 ],
       [  4.21, -11.06,  21.48,  24.33,  16.39],
       [  4.2 ,  11.31,  25.11,  19.97,   9.09],
       [  2.82,  -0.64,  23.94,  15.98,  16.23],
       [  2.61,  28.85,  13.22,   4.23,  11.99]
       ])
    names = [
        'Exxon',
        'British Telecom',
        'Dell Computer',
        'Mobil',
        'AEGON Ins. Group',
        'Vodafone Group ADR',
        'Wells Fargo',
        'Duke Energy',
        'Safeway',
        'America Online',
        'Southern Company',
        'Canon, Inc.',
        "Gen'l Re Corp.",
        'PG&E Corp.',
        'British Petroleum',
        'Honda Motor ADR',
        'Texaco',
        'Texas Utilities'
    ]
    dea = DEA(X,y)
    dea.name_units(names)
    dea.fit()
