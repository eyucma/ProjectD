"""
This module contains functions to calculate business times as a fraction of a business year
"""

from typing import Dict

import pandas as pd

from app.utils.types import Dates, ArrayLike


class DateWrapper:
    """
    Class to handle mappings of time to business time
    """

    def __init__(
        self,
        holidays: Dates,
        time_now: pd.Timestamp | str,
        time_mapping: Dict[float, float] | None = None,
    ) -> None:
        if time_mapping is None:
            self.times = {}
        else:
            self.times = time_mapping
        self.now = time_now
        self.holidays = holidays

    def update(self, ts: ArrayLike) -> None:
        """
        Updates mapping based on a list of dates (Dates)
        """
        times = pd.Series(ts)
        times = times.unique()
        for t in times:
            if t not in self.times:
                self.times[t] = self.get_time(self.now, t, holidays=self.holidays)

    def get_time(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        holidays: Dates | None = None,
    ) -> float:
        """
        Function that calculates number of business minutes from start
        to end as a fraction of entire business year
        """
        # Standard U.S. trading minutes per year
        annual_trading_minutes = 98280

        start_dt = pd.to_datetime(start).tz_localize(None)
        end_dt = pd.to_datetime(end).tz_localize(None)

        if end_dt <= start_dt:
            return 0.0

        # Create every minute between start and end (exclusive of end)
        minutes = pd.date_range(start_dt, end_dt, freq="min")

        # 1. Filter: Weekdays only (dayofweek < 5)
        minutes = minutes[minutes.dayofweek < 5]  # pylint: disable=E1101

        # 2. Filter: Market Hours (9:30 AM to 4:00 PM)
        minutes_td = minutes - minutes.normalize()
        market_open = pd.Timedelta(hours=9, minutes=30)
        market_close = pd.Timedelta(hours=16, minutes=0)
        minutes = minutes[(minutes_td >= market_open) & (minutes_td < market_close)]

        # 3. Filter: Holidays
        if holidays is not None:
            holidays_dt = pd.to_datetime(holidays).normalize()  # type: ignore # pylint: disable=E1101
            minutes = minutes[~(minutes.normalize().isin(holidays_dt))]  # type: ignore

        # Return T (total minutes / annual minutes)
        return len(minutes) / annual_trading_minutes
