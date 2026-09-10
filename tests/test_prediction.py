import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from api_config import get_match_odds, odds_to_implied_probs
from predict_today import history_before, build_dataset, per_team_long, build_match_row, predict_symmetric, train_model, FEATURES

class PredictionTests(unittest.TestCase):
    def test_odds_reversed_and_validated(self):
        data=[{'home_team':'A','away_team':'B','bookmakers':[{'markets':[{'key':'h2h','outcomes':[{'name':'A','price':2},{'name':'Draw','price':3},{'name':'B','price':4}]}]}]}]
        self.assertEqual(get_match_odds('B','A',data),{'home':4,'draw':3,'away':2})
        self.assertIsNone(get_match_odds('','B',data))
        for value in (0,1,-2,float('nan'),float('inf'),None):
            self.assertIsNone(odds_to_implied_probs({'home':value,'draw':3,'away':4}))
        self.assertAlmostEqual(sum(odds_to_implied_probs({'home':2,'draw':3,'away':4}).values()),1)

    def history(self):
        rows=[]
        for i in range(90):
            rows.append({'date':pd.Timestamp('2020-01-01')+pd.Timedelta(days=i*20),
                         'home_team':'A' if i%2 else 'B','away_team':'B' if i%2 else 'A',
                         'home_score':[2,1,0][i%3],'away_score':1,'tournament':'Friendly','neutral':1})
        return pd.DataFrame(rows)

    def test_future_results_cannot_change_prediction_features(self):
        data=self.history(); cutoff='2023-01-01'
        altered=data.copy(); altered.loc[altered.date>=cutoff,'home_score']=99
        _, a=build_dataset(history_before(data,cutoff)); _, b=build_dataset(history_before(altered,cutoff))
        self.assertEqual(a,b)
        long=per_team_long(history_before(data,cutoff))
        with patch('predict_today.get_team_id_api_football',side_effect=AssertionError('Unexpected API call')):
            row=build_match_row(long,a,'A','B',True,4,cutoff)
        self.assertEqual(list(row.columns),FEATURES)

    def test_training_and_prediction_smoke(self):
        data=self.history(); ds,elo=build_dataset(data)
        model,_,_=train_model(ds.iloc[:60],ds.iloc[60:])
        result=predict_symmetric(model,per_team_long(data),elo,'A','B','2026-01-01',True,4)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(sum(result),1,places=6)
        reverse=predict_symmetric(model,per_team_long(data),elo,'B','A','2026-01-01',True,4)
        np.testing.assert_allclose(result,reverse[::-1])

if __name__=='__main__': unittest.main()
