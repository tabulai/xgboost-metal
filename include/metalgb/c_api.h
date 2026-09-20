#pragma once
#include "xgboost/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* JSON profile of GPU command counts and elapsed seconds for this booster.
 * The returned string remains valid until the next C API call on this thread.
 * DMatrix ingestion is measured separately from the learner's commands.
 */
XGB_DLL int XGBoosterMetalProfile(BoosterHandle handle, char const** out);

/* Adapter-specific symbols outside the upstream public C header.
 * Separate split/leaf gradients are supported; distributed recovery is not.
 */
XGB_DLL int XGBoosterTrainOneIterWithSplitGrad(BoosterHandle handle, DMatrixHandle dtrain,
    int iteration, char const* split_grad, char const* split_hess,
    char const* value_grad, char const* value_hess);
XGB_DLL int XGCommunicatorSignalError(void);

#ifdef __cplusplus
}
#endif
