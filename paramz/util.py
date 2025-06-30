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
# * Neither the name of paramz.util nor the names of its
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
import warnings

def _inherit_doc(fromclass, done_classes = None):
    inherited = ''
    if done_classes is None:
        done_classes = []
    for c in fromclass.__bases__:
        if c not in done_classes:
            try:
                subs = "Inherited from {}: \n".format(c.__name__)
                subs += c.__doc__
                subs += '\n'
                inherited += subs
                done_classes.append(c)
            except TypeError as e:
                pass
        inherited += _inherit_doc(c, done_classes=done_classes)        
        
    return inherited


def _set_mem_addr(dest, src) -> None:
    """
    This function serves to replace the `.data` getter/setter that existed in
    `numpy<2` and got removed in `numpy>=2`.
    The original behavior was setting the memory address of dest to that of src.
    However, directly setting the memory address of a numpy array to the data of
    another one seems to be unwanted in `numpy>=2`, which is causing some major 
    problems here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # original
        # dest.data = src.data
        
        # take 1
        dest.data = memoryview(src)
