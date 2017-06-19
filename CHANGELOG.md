Changelog
=========

%%version%% (unreleased)
------------------------

New
~~~

- Changelog now markdown. [mzwiessele]

- Added CHANGELOG from GITCHANGELOG. [mzwiessele]

- Util for doc. [mzwiessele]

Fix
~~~

- Changing to python 3.6, supporting 3.5 still. [mzwiessele]

- Added license in manifest for package distribution. [mzwiessele]

- Load swallowed the import errors from missing packages... [mzwiessele]

- Added return for linking. [mzwiessele]

- Doc inheritance for classes. [mzwiessele]

- Doc upload. [mzwiessele]

- Updated example tests for m.phi and m.predict. [mzwiessele]

- Fixed #13 decorator package minimum required version. [mzwiessele]

- Fixed issue when loading models
  https://github.com/SheffieldML/GPy/issues/445 amend. [mzwiessele]

- Fixed issue when loading models
  https://github.com/SheffieldML/GPy/issues/445. [mzwiessele]

Other
~~~~~

- Bump version: 0.7.7 → 0.7.8. [mzwiessele]

- Bump version: 0.7.6 → 0.7.7. [mzwiessele]

- Merge pull request #19 from zhenwendai/master. [Max Zwiessele]

  add the support for Adam from climin

- Add a unittest. [Zhenwen Dai]

- Bump the version up. [Zhenwen Dai]

- Add the support of adam. [Zhenwen Dai]

- Bump version: 0.7.5 → 0.7.6. [mzwiessele]

- Bump version: 0.7.4 → 0.7.5. [mzwiessele]

- Bump version: 0.7.3 → 0.7.4. [mzwiessele]

- Merge pull request #16 from alexfeld/parallel_opt. [Max Zwiessele]

  Parallel optimization

- Fix bug in `model.optimize_restarts` that was assuming the underlying
  model has not changed. Now only searches the most recent randomized
  models and the current model. [Alex Feldstein]

- Add parallel test for `model.optimize_restarts` [Alex Feldstein]

- Fix randomize bug in parallel optimization. [Alex Feldstein]

- Fix in tests for new Optimizer.__getstate__ [Alex Feldstein]

- Fix parallel optimization. [Alex Feldstein]

- Merge pull request #15 from larroy/master. [Max Zwiessele]

  Remove nasty try except around decorator import which will cause sile…

- Remove nasty try except around decorator import which will cause
  silent unit test failing when decorator is not present. [Pedro Larroy]

- Bump version: 0.7.2 → 0.7.3. [mzwiessele]

- Bump version: 0.7.1 → 0.7.2. [mzwiessele]

- Bump version: 0.7.0 → 0.7.1. [mzwiessele]

- Merge pull request #14 from robertocalandra/master. [Max Zwiessele]

  Fixed warning

- Fixed warning. [Roberto Calandra]

- Bump version: 0.6.10 → 0.7.0. [mzwiessele]

- Bump version: 0.6.9 → 0.6.10. [mzwiessele]

- Update README.md. [Max Zwiessele]

- Bump version: 0.6.8 → 0.6.9. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Merge. [mzwiessele]

- Merge pull request #12 from sods/initialize_and_updates. [Max
  Zwiessele]

  Initialize and updates

- Bump version: 0.6.6 → 0.6.7. [mzwiessele]

- [README] updated for initialization. [mzwiessele]

- [initialize] the initialization wasn't working to set parameters.
  [mzwiessele]

- Bump version: 0.6.7 → 0.6.8. [mzwiessele]

- Bump version: 0.6.6 → 0.6.7. [mzwiessele]

- Bump version: 0.6.5 → 0.6.6. [mzwiessele]

- [fixes] added fix for fixes. [mzwiessele]

- Bump version: 0.6.4 → 0.6.5. [mzwiessele]

- [constraints] set direct tests. [mzwiessele]

- Bump version: 0.6.3 → 0.6.4. [mzwiessele]

- [constraints] set direct tests. [mzwiessele]

- [constraints] set direct tests. [mzwiessele]

- Bump version: 0.6.2 → 0.6.3. [mzwiessele]

- [paramz] set constraints directly testing. [mzwiessele]

- Bump version: 0.6.1 → 0.6.2. [mzwiessele]

- Bump version: 0.6.0 → 0.6.1. [mzwiessele]

- [indexops] directly setting the index ops now works. [mzwiessele]

- Bump version: 0.5.7 → 0.6.0. [mzwiessele]

- [py3] not supporting 3.3 any more. [mzwiessele]

- Bump version: 0.5.6 → 0.5.7. [mzwiessele]

- Bump version: 0.5.5 → 0.5.6. [mzwiessele]

- [verb opt] length of maxiters. [mzwiessele]

- Bump version: 0.5.4 → 0.5.5. [mzwiessele]

- [testing] checkgrad test changed. [mzwiessele]

- Bump version: 0.5.3 → 0.5.4. [mzwiessele]

- [testing] checkgrad test changed. [mzwiessele]

- Bump version: 0.5.2 → 0.5.3. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Bump version: 0.5.1 → 0.5.2. [mzwiessele]

- [testing] return True for all no free parameters. [mzwiessele]

- [setup] more docstringing. [mzwiessele]

- Bump version: 0.5.0 → 0.5.1. [mzwiessele]

- [rtfd] removing rtfd IT DOES NOT WORK. [mzwiessele]

- Bump version: 0.4.2 → 0.5.0. [mzwiessele]

- [rtfd] removing rtfd IT DOES NOT WORK. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- [readthedocs] weirdnessssss. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Merge pull request #11 from zhenwendai/master. [Max Zwiessele]

  numerical enhancement for lopexp transformation

- Merge branch 'master' of github.com:sods/paramz. [Zhenwen Dai]

- Merge branch 'master' of github.com:sods/paramz. [Zhenwen Dai]

- Merge with upstream. [Zhenwen Dai]

- Numerical improvement logexp. [Zhenwen Dai]

- Bug fix for negativeLopexp. [Zhenwen Dai]

- Bug fix for old caching. [Zhenwen Dai]

- Fallback to the old implementation of caching. [Zhenwen Dai]

- Bug fix for lbfgs with more than 15,000 iterations. [Zhenwen Dai]

- Bump version: 0.4.1 → 0.4.2. [mzwiessele]

- [README] changes and copyright. [mzwiessele]

- Bump version: 0.4.0 → 0.4.1. [mzwiessele]

- [README] [mzwiessele]

- [doc] updated. [mzwiessele]

- Bump version: 0.3.0 → 0.4.0. [mzwiessele]

- Merge pull request #9 from sods/initialize. [Max Zwiessele]

  [initialize] adding initialize keyword argument, so that we can leave…

- [pydot] is tested in version 2.7 but ignored in codecov, as pydot is
  not py3 compatible. [mzwiessele]

- [caching] inspect in different python versions works differently,
  changed to check 3.3 first. [mzwiessele]

- [coverage] big coverage increase and deletion of unecessery code.
  [mzwiessele]

- Merge branch 'master' into initialize. [mzwiessele]

- [caching] adding cache object, which handles on and off switching of
  caches. [mzwiessele]

- [initialize] adding initialize keyword argument, so that we can leave
  out initialization on model creation. [mzwiessele]

- [initialize] adding initialize keyword argument, so that we can leave
  out initialization on model creation. [mzwiessele]

- Merge pull request #10 from AlexGrig/opt_list_sort. [Max Zwiessele]

- FIX: Prevent selecting optimizer dependcy on iteration order of
  dictionary keys. [Alexander Grigorievskiy]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Update README.md. [Max Zwiessele]

- Bump version: 0.2.2 → 0.3.0. [mzwiessele]

- Merge branch 'master' into caching. [mzwiessele]

- Merge pull request #7 from sods/verbose. [Max Zwiessele]

  [verbose] changed verbose optimization

- [coverage] some tests for increasing coverage. [mzwiessele]

- [natgrad] deactivating for now. [mzwiessele]

- [deprecation] not covering. [mzwiessele]

- [verbose] changed verbose optimization. [mzwiessele]

- [py3] update to dict structure of py" "" [mzwiessele]

- [caching] restructured caching with decorator module. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Merge pull request #6 from sods/rprop. [Max Zwiessele]

  [rprop] added rprop optimizer and tests

- [climin] added tests for climin adadelta and rprop optimizers and
  install instr for climin. [mzwiessele]

- [stochastics] deleted and transferred to Gpy. [mzwiessele]

- [rprop] added rprop optimizer and tests. [mzwiessele]

- Bump version: 0.2.1 → 0.2.2. [mzwiessele]

- [verbose opt] printing every second now. [mzwiessele]

- Bump version: 0.2.0 → 0.2.1. [mzwiessele]

- Bump version: 0.1.6 → 0.2.0. [mzwiessele]

- [travis] source the scripts. [mzwiessele]

- [travis] source the scripts. [mzwiessele]

- [travis] source the scripts. [mzwiessele]

- [travis] script download. [mzwiessele]

- [travis] script download. [mzwiessele]

- [travis] script download. [mzwiessele]

- [travis] script download. [mzwiessele]

- [travis] script download. [mzwiessele]

- [travis] retry. [mzwiessele]

- [travis] [mzwiessele]

- Bump version: 0.1.5 → 0.1.6. [mzwiessele]

- [travis] added retries for installing conda, as it seemed to fail on
  404 errors. [mzwiessele]

- Bump version: 0.1.4 → 0.1.5. [mzwiessele]

- [gradcheck] was always checking the whole model. [mzwiessele]

- [gradcheck] was always checking the whole model. [mzwiessele]

- [constrainable] default fixed was bugged. [mzwiessele]

- [constrainable] default fixed was bugged. [mzwiessele]

- [paramz] raveled indices now easier accessible and adjusted checkgrad
  for it. [mzwiessele]

- Bump version: 0.1.3 → 0.1.4. [mzwiessele]

- Bump version: 0.1.2 → 0.1.3. [mzwiessele]

- [optimization] @zenwhendai fixes the max iterations in bfgs > 15,000.
  [mzwiessele]

- [coverage] added except clause for keyerrors. [mzwiessele]

- [dtype] in obsar with tests. [mzwiessele]

- Observable array now enforces dtype to be float only if input is
  numeric, otherwise it inherits the type. [Daniel Beck]

- Bump version: 0.1.1 → 0.1.2. [mzwiessele]

- [paramconcat] testing increase. [mzwiessele]

- [paramconcat] printing was messsed up with the length of the iop
  entries. [mzwiessele]

- [paramconcat] printing was messsed up with the length of the iop
  entries. [mzwiessele]

- [caching] added safety catches for caching to reset. [mzwiessele]

- Bump version: 0.1.0 → 0.1.1. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Merge pull request #2 from mrksr/fix_getstate_for_optimizers. [Max
  Zwiessele]

  getstate should return a dictionary

- Check for correct equality in the testcase. [Markus Kaiser]

- Getstate should return a dictionary. [Markus Kaiser]

- [examples] added one weight per dimension for ridge regression and
  added in basis functions. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Bump version: 0.0.35 → 0.1.0. [mzwiessele]

- [constrainable] fixing now does not delete previous constraints, so
  unfix will restore previous constraint. [mzwiessele]

- [transformations] coverage increase. [mzwiessele]

- Bump version: 0.0.34 → 0.0.35. [mzwiessele]

- [tests] coverage up and extras reqs added. [mzwiessele]

- Bump version: 0.0.33 → 0.0.34. [mzwiessele]

- [load] loading of pickle fallback to non c-code. [mzwiessele]

- [load] loading of pickle fallback to non c-code. [mzwiessele]

- Bump version: 0.0.32 → 0.0.33. [mzwiessele]

- [param] gradient setting was not checking if the array is not set yet.
  [mzwiessele]

- Bump version: 0.0.31 → 0.0.32. [mzwiessele]

- [paramz] undo ccontig. [mzwiessele]

- Bump version: 0.0.30 → 0.0.31. [mzwiessele]

- [readme.md] added. [mzwiessele]

- Bump version: 0.0.29 → 0.0.30. [mzwiessele]

- Bump version: 0.0.28 → 0.0.29. [mzwiessele]

- [readme] etf's with pypi morooororororoeee. [mzwiessele]

- [readme] wtf's. [mzwiessele]

- Bump version: 0.0.27 → 0.0.28. [mzwiessele]

- [pypi] rst not rendering, holy moly this is annoying. [mzwiessele]

- Bump version: 0.0.26 → 0.0.27. [mzwiessele]

- [param] preventing adding parameters which have not the same shape as
  their realshape. [mzwiessele]

- Bump version: 0.0.25 → 0.0.26. [mzwiessele]

- [paramz] c-cont message enhanced and found another error which is
  reported to the user. [mzwiessele]

- Bump version: 0.0.24 → 0.0.25. [mzwiessele]

- [examples] added tests for ridge regression. [mzwiessele]

- [optimization] coverage increase for scg. [mzwiessele]

- Bump version: 0.0.23 → 0.0.24. [mzwiessele]

- [optimization] test coverage. [mzwiessele]

- Bump version: 0.0.22 → 0.0.23. [mzwiessele]

- [html] printing of param class. [mzwiessele]

- Bump version: 0.0.21 → 0.0.22. [mzwiessele]

- [pypi] release only on tag. [mzwiessele]

- Bump version: 0.0.20 → 0.0.21. [mzwiessele]

- [imports] cPickle. [mzwiessele]

- [imports] [mzwiessele]

- Bump version: 0.0.19 → 0.0.20. [mzwiessele]

- [optimizer] not saving x_init, as it clutters pickling. [mzwiessele]

- [examples] added ridge regression. [mzwiessele]

- Merge branch 'master' of github.com:sods/paramz. [mzwiessele]

- Bump version: 0.0.18 → 0.0.19. [mzwiessele]

- Bump version: 0.0.17 → 0.0.18. [mzwiessele]

- [regexp] matching setting getting and concatenation. [mzwiessele]

- Bump version: 0.0.16 → 0.0.17. [mzwiessele]

- [regexp] matching setting getting and concatenation. [mzwiessele]

- [regexp] matching setting getting and concatenation. [mzwiessele]

- Bump version: 0.0.15 → 0.0.16. [mzwiessele]

- [regexp] matching setting getting and concatenation. [mzwiessele]

- Bump version: 0.0.14 → 0.0.15. [mzwiessele]

- [readme] was not being converted to rst. [mzwiessele]

- Bump version: 0.0.13 → 0.0.14. [mzwiessele]

- [readme] was not being converted to rst. [mzwiessele]

- Bump version: 0.0.12 → 0.0.13. [mzwiessele]

- [readme] was not being converted to rst. [mzwiessele]

- Bump version: 0.0.11 → 0.0.12. [mzwiessele]

- [coverage] adjustments. [mzwiessele]

- [sdist] is not uploaded, why? [mzwiessele]

- [sdist] is not uploaded, why? [mzwiessele]

- [sdist] is not uploaded, why? [mzwiessele]

- Bump version: 0.0.10 → 0.0.11. [mzwiessele]

- [README] added to manifest. [mzwiessele]

- Bump version: 0.0.9 → 0.0.10. [mzwiessele]

- [README] badges changed for sods. [mzwiessele]

- Rename README -> README.md. [mzwiessele]

- Bump version: 0.0.8 → 0.0.9. [mzwiessele]

- [python3] updates. [mzwiessele]

- [doc] update and test coverage increase. [mzwiessele]

- Bump version: 0.0.7 → 0.0.8. [mzwiessele]

- [ordering] different in python versions... [mzwiessele]

- [python3] transferred. [mzwiessele]

- Bump version: 0.0.6 → 0.0.7. [mzwiessele]

- [optimization] more warnings and option checking. [mzwiessele]

- [optimization] test coverage increased, testing optimizers.
  [mzwiessele]

- Bump version: 0.0.5 → 0.0.6. [mzwiessele]

- [unpickling] added constraints and constraint dict to setstate.
  [mzwiessele]

- Bump version: 0.0.4 → 0.0.5. [mzwiessele]

- [README.md] -> README. [mzwiessele]

- [README] README.md -> README. [mzwiessele]

- Bump version: 0.0.3 → 0.0.4. [mzwiessele]

- Bump version: 0.0.2 → 0.0.3. [mzwiessele]

- [version] bump. [mzwiessele]

- Update parameterized.py. [Max Zwiessele]

- [printing] updates on the printing, html todo. [mzwiessele]

- Merge pull request #1 from mzwiessele/mzwiessele-patch-1. [Max
  Zwiessele]

  Update parameterized.py

- Update param.py. [Max Zwiessele]

- Update parameterized.py. [Max Zwiessele]

- [printing] todo: html. [mzwiessele]

- [examples] added to setup. [mzwiessele]

- [paramz] printing. [mzwiessele]

- Bump version: 0.0.2 → 0.0.3. [mzwiessele]

- [testing] some extension settings. [mzwiessele]

- [python3] reduce import. [mzwiessele]

- [constrainable] as its own class. [mzwiessele]

- [constrainable] as its own class. [mzwiessele]

- [docs] [mzwiessele]

- [docs] [mzwiessele]

- Bump version: 0.0.1 → 0.0.2. [mzwiessele]

- [travis] deploy. [mzwiessele]

- Bump version: 0.0.0 → 0.0.1. [mzwiessele]

- [setup] [mzwiessele]

- Bump version: 0.0.0 → 0.0.1. [mzwiessele]

- [setup] [mzwiessele]

- [setup] [mzwiessele]

- Bump version: 0.0.0 → 0.0.1. [mzwiessele]

- [badges] and right setup for uploading dist. [mzwiessele]

- [travis] deployment. [mzwiessele]

- [readme] badges. [mzwiessele]

- Bump version: 0.0.2 → 0.0.3. [mzwiessele]

- Revert "Bump version: 0.0.2 → 0.0.3" [mzwiessele]

  This reverts commit 5d76028695430683a001eac02226adbc78a17346.

- Revert "Bump version: 0.0.3 → 0.0.4" [mzwiessele]

  This reverts commit 7de71431f1e45fc977114d0774b111dfa3d7ea9a.

- Bump version: 0.0.3 → 0.0.4. [mzwiessele]

- Bump version: 0.0.2 → 0.0.3. [mzwiessele]

- [testing] bumpversion. [mzwiessele]

- Bump version: 0.0.1 → 0.0.2. [mzwiessele]

- [doc] added. [mzwiessele]

- Bump version: 0.0.0 → 0.0.1. [mzwiessele]

- [obsar] pargma cover ignores. [mzwiessele]

- [obsar] pargma cover ignores. [mzwiessele]

- [dictionary] ordering changed behaviour from py2 to 3. [mzwiessele]

- [travis] included. [mzwiessele]

- [paramz] restructured a lot and made it runnable (tests run through)
  see GPy paramz branch for inclusion. [mzwiessele]

- [initial] copied files over now to making it work. [mzwiessele]

- [testing] added travis files and setup. [mzwiessele]

- Initial commit. [Max Zwiessele]


