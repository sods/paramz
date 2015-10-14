'''
Copyright (c) 2015, Max Zwiessele
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of paramz nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import unittest, numpy as np
from ..core.lists_and_dicts import ArrayList, IntArrayDict, ObserverList
from ..core.observable_array import ObsAr
from ..parameterized import Parameterized
        
class Test(unittest.TestCase):


    def setUp(self):
        
        pass


    def tearDown(self):
        pass


    def testArrayList(self):
        t = []
        a = ArrayList()
        for r in range(3,6):
            _ = np.random.normal(0,1,r)
            a.append(_)
            t.append(_)
        r = np.random.normal()
        self.assertRaises(ValueError, t.__contains__, r)
        self.assertFalse(a.__contains__(r))
        self.assertRaises(ValueError, t.index, r)
        self.assertRaises(ValueError, a.index, r)

        r = a[2]
        self.assertTrue(a.__contains__(r))
        self.assertEqual(a.index(r), 2)

    def testPrintObserverListObsAr(self):
        o = ObserverList()
        test1 = ObsAr(np.array([1]))
        o.add(1, test1, None)
        #tstr = ("; to 'ObsAr' at " + hex(id(test1)) + ">, None)]")
        #self.assertSequenceEqual(repr(o)[-len(tstr):], tstr)
        #tstr = '[(1, <weakref at '
        #self.assertSequenceEqual(repr(o)[:len(tstr)], tstr)
        print(o)

    def testPrintObserverListParameterized(self):
        o = ObserverList()
        test2 = Parameterized()
        o.add(1, test2, None)
        print(o)
        class O(object):
            pass
        test3 = O()
        o.add(3, test3, None)
        print(o)

    def testPrintObserverListObj(self):
        o = ObserverList()
        class O(object):
            pass
        test3 = O()
        o.add(3, test3, None)
        print(o)

    def testPrintPriority(self):
        o = ObserverList()
        class O(object):
            pass
        t = [O() for _ in range(4)]
        o.add(5, t[0], None)
        o.add(3, t[1], None)
        o.add(3, t[2], None)
        o.add(2, t[3], None)
        print(o)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()