"""
This module contains functions to calculate business times as a fraction of a business year
"""

import pandas as pd

from app.utils.types import Dates


def get_decimal_T(start: pd.Timestamp|str, end: pd.Timestamp|str, holidays: Dates|None = None):
    # Standard U.S. trading minutes per year
    ANNUAL_TRADING_MINUTES = 98280

    start_dt = pd.to_datetime(start).tz_localize(None)
    end_dt = pd.to_datetime(end).tz_localize(None)

    if end_dt <= start_dt:
        return 0.0

    # Create every minute between start and end (exclusive of end)
    minutes = pd.date_range(start_dt, end_dt, freq="min")

    # 1. Filter: Weekdays only (dayofweek < 5)
    minutes = minutes[minutes.dayofweek < 5]

    # 2. Filter: Market Hours (9:30 AM to 4:00 PM)
    minutes_td = minutes - minutes.normalize()
    market_open = pd.Timedelta(hours=9, minutes=30)
    market_close = pd.Timedelta(hours=16, minutes=0)
    minutes = minutes[(minutes_td >= market_open) & (minutes_td < market_close)]

    # 3. Filter: Holidays
    if holidays is not None:
        holidays_dt = pd.to_datetime(holidays)
        holidays_dt = pd.DatetimeIndex(holidays_dt).normalize()

        minutes = minutes[~(minutes.normalize().isin(holidays_dt))]  # type: ignore

    # Return T (total minutes / annual minutes)
    return len(minutes) / ANNUAL_TRADING_MINUTES

class TimeWrapper:
    def __init__(
        self, holidays: Dates, time_now:pd.Timestamp|str, time_mapping:dict={}
    ) -> None:
        self.times = time_mapping
        self.time_now = time_now
        self.holidays = holidays

    def update(self, Ts) -> None:
        times = Ts.unique()
        for t in times:
            if t not in self.times:
                self.times[t] = self.get_decimal_T(self.time_now, t)

    def get_decimal_T(self, start, end):
        # Standard U.S. trading minutes per year
        ANNUAL_TRADING_MINUTES = 98280

        start_dt = pd.to_datetime(start).tz_localize(None)
        end_dt = pd.to_datetime(end).tz_localize(None)

        if end_dt <= start_dt:
            return 0.0

        # Create every minute between start and end (exclusive of end)
        minutes = pd.date_range(start_dt, end_dt, freq="min")

        # 1. Filter: Weekdays only (dayofweek < 5)
        minutes = minutes[minutes.dayofweek < 5]

        # 2. Filter: Market Hours (9:30 AM to 4:00 PM)
        minutes_td = minutes - minutes.normalize()
        market_open = pd.Timedelta(hours=9, minutes=30)
        market_close = pd.Timedelta(hours=16, minutes=0)
        minutes = minutes[(minutes_td >= market_open) & (minutes_td < market_close)]

        # 3. Filter: Holidays
        if self.holidays is not None:
            holidays_norm = pd.to_datetime(self.holidays).normalize()
            minutes = minutes[~minutes.normalize().isin(holidays_norm)]

        # Return T (total minutes / annual minutes)
        return len(minutes) / ANNUAL_TRADING_MINUTES


us_holidays = USFederalHolidayCalendar().holidays(
    start=time_now, end=X["expiration"].max()
)

tm = time_mapper(holidays=us_holidays, time_now=time_now)
tm.update(X["expiration"].unique())
X["T"] = X["expiration"].map(tm.times)
