#===============================================================================
# Copyright (c) 2013-2015, Max Zwiessele
#
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
# * Neither the name of paramax nor the names of its
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

import numpy

class GDUpdateRule():
    _gradnat = None
    _gradnatold = None
    def __init__(self, initgrad, initgradnat=None):
        self.grad = initgrad
        if initgradnat:
            self.gradnat = initgradnat
        else:
            self.gradnat = initgrad
        # self.grad, self.gradnat
    def _gamma(self):
        raise NotImplemented("""Implement gamma update rule here, 
        you can use self.grad and self.gradold for parameters, as well as
        self.gradnat and self.gradnatold for natural gradients.""")
    def __call__(self, grad, gradnat=None, si=None, *args, **kw):
        """
        Return gamma for given gradients and optional natural gradients
        """
        if not gradnat:
            gradnat = grad
        self.gradold = self.grad
        self.gradnatold = self.gradnat
        self.grad = grad
        self.gradnat = gradnat
        self.si = si
        return self._gamma(*args, **kw)

class FletcherReeves(GDUpdateRule):
    '''
    Fletcher Reeves update rule for gamma
    '''
    def _gamma(self, *a, **kw):
        tmp = numpy.dot(self.grad.T, self.gradnat)
        if tmp:
            return tmp / numpy.dot(self.gradold.T, self.gradnatold)
        return tmp

class PolakRibiere(GDUpdateRule):
    '''
    Fletcher Reeves update rule for gamma
    '''
    def _gamma(self, *a, **kw):
        tmp = numpy.dot((self.grad - self.gradold).T, self.gradnat)
        if tmp:
            return tmp / numpy.dot(self.gradold.T, self.gradnatold)
        return tmp
