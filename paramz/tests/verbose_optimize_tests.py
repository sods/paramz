#===============================================================================
# Copyright (c) 2016, Max Zwiessele
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of paramz.tests.verbose_optimize_tests nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

import unittest
from paramz.optimization.verbose_optimization import VerboseOptimization
from paramz.optimization.optimization import opt_bfgs

class Test(unittest.TestCase):
    def setUp(self):
        class Stub(object):
            obj_grads = [10,0]
            def add_observer(self, m, f):
                self.obs = m
                self.obs_f = f
            def objective_function(self):
                return 10
                
        self.vo = VerboseOptimization(Stub(), opt_bfgs(), -10, verbose=True)

    def test_timestrings(self):
        self.vo.print_out(0)
        self.assertEqual(self.vo.timestring, '00s00')

        self.vo.print_out(10.2455)
        self.assertEqual(self.vo.timestring, '10s24')

        self.vo.print_out(120)
        self.assertEqual(self.vo.timestring, '02m00s00')

        self.vo.print_out(60*60+120+12.2455)
        self.assertEqual(self.vo.timestring, '01h02m12')

        self.vo.print_out(2*3600*24+60*60+120+12.2455)
        self.assertEqual(self.vo.timestring, '02d01h02')

    def test_finish(self):
        self.assertEqual(self.vo.status, 'running')
        
        