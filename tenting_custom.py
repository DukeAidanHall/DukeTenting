"""
Tenting Scheduler

If you are a human see lines 15-23 and lines 172->
"""

from dataclasses import dataclass
from datetime import time
from typing import List
import pandas as pd
import numpy as np
import csv


## ==============================
#Tent Scheduling Algorithm Control Zone
#You can alter these coefficients to change the algorithm's behavior and focus on different factors.

avaliable = 100 #keep positive (incr -->increases importance of being avaliable)
continuity = 20 #keep positive (incr --> increases importance of continuity of shifts)
hours_dec = -0.5 #keep negative and small (decr ---> increases importance of current total hours at any point)
focus_inc = 0.1 #keep positive and kinda small (incr ---> increases importance of slowly increasing focus variable)
deficit_value = 4 #keep positive (incr --> increases importance of minimizing max-min hours)
START_SLOT_INDEX =  0    # inclusive (how many hours you want to miss on the front end * 2)
END_SLOT_INDEX   =  336  # exclusive ( 336 - how many hours you want to skip on the back end * 2)
OFFSET_ARRAY = [0,0,0,0,0,0,0,0,0,0,0,0] # positive values only, this should have the excess hours of each person in order
SETnightNum = 6
SETdayNum = 1

## ==============================



# ======================
# CONSTANTS
# ======================

DAY_ORDER = [
    "Sunday", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday"
]

# ======================
# DATA STRUCTURES
# ======================

@dataclass(frozen=True)
class Slot:
    day: int           # 0 = Monday ... 6 = Sunday
    row: int           # index in time list
    start: time        # start time
    label: str         # "0:00-0:30"
    index: int         # global chronological index

# ======================
# TIME HELPERS
# ======================

def parse_hhmm(s: str) -> time:
    h, m = map(int, s.strip().split(":"))
    return time(h, m)

def parse_timerange_start(timerange: str) -> time:
    return parse_hhmm(timerange.split("-")[0])

def is_sleep_window(t: time) -> bool:
    """True if 11pm–7am"""
    return t >= time(23, 0) or t < time(7, 0)

def classify_day_or_night(t: time) -> str:
    return "day" if t >= time(7, 0) else "night"

def as_bool_cell(x) -> bool:
    """TRUE/FALSE cell → bool"""
    return str(x).strip().lower() == "true"

# ======================
# CORE LOADER
# ======================

def load_tenting_data(input_csv: str, data_zone_csv: str):
    """
    Returns:
      people: List[str]
      slots:  List[Slot]
      busy:   busy[p][day][row] -> bool
    """

    # -------- Load people --------
    dz = pd.read_csv(data_zone_csv)
    if "Names" not in dz.columns:
        raise ValueError("Data Zone CSV must contain 'Names' column")

    people = dz["Names"].astype(str).tolist()
    n_people = len(people)

    # -------- Load availability grid --------
    grid = pd.read_csv(input_csv, header=None, dtype=str)

    # Extract time rows
    time_rows = [
        i for i in range(1, len(grid))
        if "-" in str(grid.iloc[i, 0])
    ]

    time_labels = [grid.iloc[i, 0] for i in time_rows]
    n_times = len(time_labels)

    # Validate structure
    if (grid.shape[1] - 1) % 7 != 0:
        raise ValueError("Expected 7 columns per person")

    # busy[p][day][row]
    busy = [[[False for _ in range(n_times)] for _ in range(7)]
            for _ in range(n_people)]

    data = grid.iloc[time_rows, 1:]

    for p in range(n_people):
        block = data.iloc[:, p * 7:(p + 1) * 7]
        for d in range(7):
            for r in range(n_times):
                busy[p][d][r] = as_bool_cell(block.iloc[r, d])

    # -------- Build slots --------
    slots: List[Slot] = []
    idx = 0

    for d in range(7):
        for r, label in enumerate(time_labels):
            slots.append(
                Slot(
                    day=d,
                    row=r,
                    start=parse_timerange_start(label),
                    label=label,
                    index=idx
                )
            )
            idx += 1

    return people, slots, busy


def write_output_csv(output_path: str, slots, people, assigned):
    """
    Writes the tenting schedule in the same format as the Automation Input CSV.
    """

    n_people = len(people)
    n_days = 7
    n_rows_per_day = 48

    # Header row
    header = ["Time"]
    for _ in range(n_people):
        header.extend(DAY_ORDER)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # For each time row (0..47)
        for row in range(n_rows_per_day):
            time_label = slots[row].label
            out_row = [time_label]

            # For each person, write 7 day values
            for p in range(n_people):
                for d in range(n_days):
                    out_row.append(
                        "TRUE" if assigned[p][d][row] else "FALSE"
                    )

            writer.writerow(out_row)


    
## ==============================
#Everything above basically just sets up input/ouput. Below is core scheduling logic.
## ==============================



def previousAssigned(p, slot, num):
    if (slot.row - num < 0):
        return assigned[p][slot.day-1][48 + (slot.row - num)]
    else:
        return assigned[p][slot.day][slot.row - num]
        
def findDayValue(slot, p, score, focus, assigned, hours):
    value = 0

    if not busy[p][slot.day][slot.row]:
        value += avaliable
    
    v8 = previousAssigned(p, slot, 8)
    v7 = previousAssigned(p, slot, 7)
    v6 = previousAssigned(p, slot, 6)
    v5 = previousAssigned(p, slot, 5)
    v4 = previousAssigned(p, slot, 4)
    v3 = previousAssigned(p, slot, 3)
    v2 = previousAssigned(p, slot, 2)
    v1 = previousAssigned(p, slot, 1)
    if v8 & v7 & v6 & v5 & v4 & v3 & v2 & v1:
        value += continuity * -9999 #1/8
    elif v7 & v6 & v5 & v4 & v3 & v2 & v1:
        value += continuity * -9999 #2/8
    elif v6 & v5 & v4 & v3 & v2 & v1:
        value += continuity * 3/8
    elif v5 & v4 & v3 & v2 & v1:
        value += continuity * 4/8
    elif v4 & v3 & v2 & v1:
        value += continuity * 5/8
    elif v3 & v2 & v1:
        value += continuity * 6/8
    elif v2 & v1:
        value += continuity * 7/8
    elif v1:
        value += continuity
    
    value += hours[p] * hours_dec

    value += focus[p] * focus_inc

    return value


##def findNightValue(slots, upper, lower, p, score, focus, assigned, hours):
    # HARD CONSTRAINT: must be free for entire night
    for i in range(lower, upper + 1):
        if busy[p][slots[i].day][slots[i].row]:
            return -1e9  # effectively disqualify

    value = 0

    # Fully available → full availability reward
    value += avaliable

    value += hours[p] * hours_dec
    value += focus[p] * focus_inc

    return value

def in_bounds(day, row):
    idx = day * 48 + row
    return START_SLOT_INDEX <= idx < END_SLOT_INDEX

def findNightValue(slots, upper, lower, p, score, focus, assigned, hours):
    value = 0
    
    availability = 0
    for row in range(lower, upper + 1):
       if not busy[p][slot.day][row]:
        availability += 1
    value += (availability / (upper - lower + 1)) * avaliable
    
    value += hours[p] * hours_dec

    value += focus[p] * focus_inc

    return value

if __name__ == "__main__":



    people, slots, busy = load_tenting_data(
        "Tenting_2026_HAJA - Automation Input.csv",
        "Tenting_2026_HAJA - Data Zone.csv"
    )
    print("----------------------")
    nightNum = SETnightNum
    dayNum = SETdayNum
    newNight = True
    maxDeficit = 999
    trialNum = 1
    score = [0 for _ in people]
    focus = [0 for _ in people]
    hours = [0 for _ in people]
    hours = hours + OFFSET_ARRAY
    assigned = [
        [
            [False for _ in range(48)]
            for _ in range(7)
        ]
        for _ in range(12)
    ]

    for slot in slots:
        
        for p, name in enumerate(people):
            if not busy[p][slot.day][slot.row]:
                score[p] = score[p] + 1
    while maxDeficit > 7:
       hours = [0 for _ in people]
       assigned = [
        [
            [False for _ in range(48)]
            for _ in range(7)
        ]
        for _ in range(12)
       ]
       for slot in slots:
           if slot.index < START_SLOT_INDEX or slot.index >= END_SLOT_INDEX:
               continue
           highest_focus = max(focus)
           values = [0 for _ in people]
           if slot.day == 0 or slot.day == 6:
              if slot.row >= 14 and slot.row <=45: #4
                newNight = True
                for p, name in enumerate(people):
                    value = findDayValue(slot, p, score, focus, assigned, hours)
                    values[p] = value
                top_indices = np.argsort(values)[-dayNum:][::-1]
                for i in top_indices:
                    if in_bounds(slot.day, slot.row):
                        assigned[i][slot.day][slot.row] = True
                        hours[i] += 0.5

              else:
                   if newNight:
                       newNight = False
                       upper = 13
                       lower = 5
                       for p, name in enumerate(people):
                          value = findNightValue(slots, upper, lower, p, score, focus, assigned, hours)
                          values[p] = value
                       top_indices = np.argsort(values)[-nightNum:][::-1] ##nightNum highest scoring indices
                       fillArray = []
                       for i in top_indices:
                               for row in range(lower,upper+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5

                               fillArray.append(hours[i])
                       top_fill_indices = np.argsort(fillArray)[-dayNum:][::-1]
                       if slot.day != 0:
                           for i in top_fill_indices:
                               for row in range(46,47+1):
                                   if in_bounds(slot.day - 1, row):
                                       assigned[i][slot.day - 1][row] = True
                                       hours[i] += 0.5

                               for row in range(0,4+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5

                       else: 
                           for i in top_fill_indices: 
                               for row in range(0,4+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5

           else:
               if slot.row >= 14 and slot.row <=45: #1
                   newNight = True
                   for p, name in enumerate(people):
                       value = findDayValue(slot, p, score, focus, assigned, hours)
                       values[p] = value
                   top_indices = np.argsort(values)[-dayNum:][::-1]
                   for i in top_indices:
                       if in_bounds(slot.day, slot.row):
                           assigned[i][slot.day][slot.row] = True
                           hours[i] += 0.5

               else:
                   if newNight:
                       newNight = False
                       upper = 13
                       lower = 2
                       for p, name in enumerate(people):
                           value = findNightValue(slots, upper, lower, p, score, focus, assigned, hours)
                           values[p] = value
                       top_indices = np.argsort(values)[-nightNum:][::-1]
                       fillArray = []
                       for i in top_indices:
                               for row in range(lower,upper+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5

                               fillArray.append(hours[i])
                       top_fill_indices = np.argsort(fillArray)[-dayNum:][::-1]
                       if slot.day != 0:
                           for i in top_fill_indices:
                               for row in range(46,47+1):
                                   if in_bounds(slot.day - 1, row):
                                       assigned[i][slot.day - 1][row] = True
                                       hours[i] += 0.5

                               for row in range(0,1+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5

                       else:
                           for i in top_fill_indices: 
                               for row in range(0,1+1):
                                   if in_bounds(slot.day, row):
                                       assigned[i][slot.day][row] = True
                                       hours[i] += 0.5
 


                       


       maxDeficit = max(hours) - min(hours)
       lowestHourIndex = np.argmin(hours)
       focus[lowestHourIndex] = focus[lowestHourIndex] + 0.1
       print(f"Trial {trialNum}, Max Deficit: {maxDeficit}")
       trialNum += 1
       print("Each person's hours: ")
       for i in range(len(hours)):
           print(f"{people[i]}: {(hours[i]-OFFSET_ARRAY[i]):.1f} hrs")
       print("----------------------")
       print("Total hours: ", sum(hours))
       print("----------------------")
       
    print("Min Deficit achieved! Writing to Tenting_2026_HAJA_Output.csv")
    forced_counts = [0 for _ in people]
    for p, name in enumerate(people):
        for d in range(7):
           for r in range(48):
               if assigned[p][d][r] and busy[p][d][r]:
                   forced_counts[p] += 1

    for p, name in enumerate(people):
       print(f"{name}: {forced_counts[p]} forced slots")
    write_output_csv(
    "Tenting_2026_HAJA_Output.csv",
    slots,
    people,
    assigned
    )

    print("Forced (busy) assignments per person:")
    print("Done!")

#for d in range(7):
#    print(f"\nDay {d} ({DAY_ORDER[d]}):")
#    for s in range(48):
#        count = sum(
#            1
#            for p in range(len(people))
#            if assigned[p][d][s]
#        )
#        print(f"  Slot {s:02d}: {count} people")
