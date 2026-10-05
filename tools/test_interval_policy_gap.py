import itertools
import unittest

from interval_policy_gap import certify, verify


def value_2x2(a):
    strategies = [0.0, 1.0]
    denominator = a[0] - a[1] - a[2] + a[3]
    if denominator:
        cross = (a[3] - a[2]) / denominator
        if 0 <= cross <= 1:
            strategies.append(cross)
    return max(min(p*a[0]+(1-p)*a[2], p*a[1]+(1-p)*a[3]) for p in strategies)


def matrix(intervals):
    return {'actions': [[0, 1], [0, 1]], 'cells': [{'lo': lo, 'hi': hi} for lo, hi in intervals]}


class CorrelatedPolicyGap(unittest.TestCase):
    def test_unknown_saddle_value_cancels_without_solving_it(self):
        m = matrix([(.3,.7),(.8,1),(0,.2),(0,1)])
        for side in [0,1]:
            c = certify(m,side,[1,0])
            self.assertLessEqual(c['loss_upper'],1e-9)
            self.assertTrue(verify(m,c))
            self.assertGreater(certify(m,side,[0,1])['loss_upper'],.1)

    def test_unknown_payoffs_do_not_produce_false_optimality(self):
        for side in [0,1]:
            self.assertEqual(certify(matrix([(0,1)]*4),side,[1,0])['loss_upper'],1)

    def test_exact_game_losses_are_enclosed_across_interval_boxes(self):
        for intervals in [[(.3,.7),(.8,1),(0,.2),(0,1)],[(0,1)]*4,
                          [(.4,.8),(.1,.3),(.2,.9),(.3,.6)],[(1,1),(0,0),(0,0),(1,1)]]:
            m=matrix(intervals)
            for side in [0,1]:
                for probability in [0,.25,.5,.75,1]:
                    p=[probability,1-probability]
                    c=certify(m,side,p)
                    self.assertTrue(verify(m,c))
                    for a in itertools.product(*[(lo,(lo+hi)/2,hi) for lo,hi in intervals]):
                        value=value_2x2(a)
                        if side==0:
                            guarantee=min(p[0]*a[0]+p[1]*a[2],p[0]*a[1]+p[1]*a[3])
                            loss=value-guarantee
                        else:
                            guarantee=1-max(p[0]*a[0]+p[1]*a[1],p[0]*a[2]+p[1]*a[3])
                            loss=1-value-guarantee
                        self.assertLessEqual(loss,c['loss_upper']+1e-12)

    def test_checker_rejects_tampered_bound_and_witness(self):
        m=matrix([(0,1)]*4)
        c=certify(m,0,[1,0]);c['loss_upper']=0
        with self.assertRaises(AssertionError):verify(m,c)
        c=certify(m,0,[1,0]);c['difference_upper_bounds'][0]+=1
        with self.assertRaises(AssertionError):verify(m,c)


if __name__=='__main__':unittest.main()
