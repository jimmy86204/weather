import pandas as pd
import numpy as np

oof_model1 = pd.read_parquet('./output/model1/oof.parq')
oof_model1['index'] = (oof_model1.DateTime.dt.strftime('%Y%m%d%H%M') + oof_model1.LocationCode.apply(lambda x: str(x).zfill(2))).astype(np.int64)
model1_sub = pd.read_csv('./output/model1/sub.csv')

oof_model2 = pd.read_csv('./output/model2/oof.csv')
model2_sub = oof_model2[oof_model2.set == 'test'].reset_index(drop=True)
oof_model2 = oof_model2[oof_model2.set == 'val'].reset_index(drop=True)

res = oof_model1.merge(
    oof_model2[['index', 'power']], on=['index'], how='inner'
)

best_score = np.inf
best_w = None
for w in np.arange(0, 1.1, 0.1):

    curr_score = ((w * res.preds.clip(0, None) + (1-w) * res.power.clip(0, None)) - res['Power(mW)']).abs().mean()
    if curr_score < best_score:
        best_score = curr_score
        best_w = w
    print(f'當前model1權重: {w}, 平均誤差分數: {curr_score}')

print(f'最佳權重為: model1: {best_w}, model2: {1-best_w}, 最佳平均誤差分數為: {best_score}')
model2_sub = model2_sub[['index', 'power']].rename(columns={'index': '序號', 'power': '答案_2'})
model1_sub = model1_sub.merge(
    model2_sub, on=['序號'], how='left'
)
model1_sub['答案'] = best_w * model1_sub['答案'].clip(0, None) + (1 - best_w) * model1_sub['答案_2'].clip(0, None)
model1_sub.drop(columns=['答案_2']).to_csv('./output/sub.csv', index=False)