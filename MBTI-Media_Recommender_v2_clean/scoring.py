import pandas as pd
import numpy as np

def add_weighted_rating(content: pd.DataFrame, q: float = 0.70):
    """
    Add IMDB-style weighted rating and Z-score columns.

    Parameters
    ----------
    content : pd.DataFrame
        콘텐츠 메타데이터(DataFrame).
        반드시 'Rating Value', 'Rating Count' 컬럼을 포함해야 함.
    q : float, default=0.70
        m(최소 투표 수) 계산을 위한 분위수(quantile).
        - 0.70 → 상위 30% 이상의 투표수를 가진 작품을 기준으로 가중치 부여
        - 값이 클수록 더 많은 작품이 평균 C 값의 영향을 크게 받음.

    Formula
    -------
    WeightedRating (WR) = (v / (v + m)) * R + (m / (v + m)) * C
        R : 작품의 평균 평점
        v : 작품의 평점 참여 수
        C : 전체 콘텐츠 평균 평점
        m : 평점 참여 수 기준 하위 q 분위수

    Returns
    -------
    content : pd.DataFrame
        WeightedRating 및 Z-score(WeightedRatingZ)가 추가된 DataFrame.
    """
    # 안전 캐스팅 및 기본 통계
    v = content["Rating Count"].astype(float)
    R = content["Rating Value"].astype(float)
    C = R.mean()
    m = v.quantile(q)

    # IMDB 방식 가중 평점 계산
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    content["WeightedRating"] = wr

    # 표준화 점수(Z-score) → 추천에서 직접 사용
    content["WeightedRatingZ"] = (wr - wr.mean()) / (wr.std() + 1e-8)

    return content
