import argparse
from operator import itemgetter # 导入定位的头方便定位按照哪里排序
import csv

#####################################config##########################################
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=int,
        default=0,
        choices=[0, 1],
        help=
        "0:e2e_task 1:vec_aicpu_task , choices=[0,1],default:0"
    )
    parser.add_argument("--summary_csv",default=None,help="op_summary_x_x_x.csv")
    args = parser.parse_args()
    return args
args = get_args()
#####################################config##########################################


Task_start_time=0.0
Task_duration_time=0.0
Task_end_time=0.0
Mac_time=0.0
All_task_kernel_time=0.0
Max_task_end_time=0.0
ind=0
ind_mac=0

#read op_summary_x_x_x.csv
if args.summary_csv is not None:
    f = csv.reader(open(args.summary_csv,'r'))

    #record all/vec+aicpu tasks
    table=[]
    for i in f:
        if ind == 0 :
            ind=i.index("Task Start Time")
            ind_mac=i.index("mac_time(us)")
            continue
        try:
            if i[ind_mac] == "N/A":
                Mac_time=0.0
            else:
                Mac_time=float(i[ind_mac])
            Task_start_time=float(i[ind])
            Task_duration_time=float(i[ind+1])*1000
            Task_end_time=Task_start_time+Task_duration_time
            if args.type == 1 and Mac_time > 0:
                continue
            All_task_kernel_time=All_task_kernel_time+Task_duration_time
            table.append([Task_start_time,Task_duration_time,Task_end_time])
        except:
            print(args.summary_csv,i)
    table_sorted = sorted(table,key=itemgetter(0),reverse=False)

    #compute parallelism time
    for j in range(1,len(table_sorted)):
        before_task_start_time=table_sorted[j-1][0]
        before_task_duration_time=table_sorted[j-1][1]
        before_task_end_time=table_sorted[j-1][-1]
        cur_task_start_time=table_sorted[j][0]
        cur_task_duration_time=table_sorted[j][1]
        cur_task_end_time=table_sorted[j][-1]
        if before_task_end_time <= cur_task_start_time:
            if  cur_task_start_time >= Max_task_end_time:
                #  |---------------task j-1-------------------|    |-----task j---------------------------|
                #  before_task_start_time   before_task_end_time    cur_task_start_time   cur_task_end_time
                #                            Max_task_end_time
                continue
            elif cur_task_start_time <= Max_task_end_time and cur_task_end_time >= Max_task_end_time:
                #  |---------------task j-1-------------------|    |-----task j---------------------------|
                #  before_task_start_time   before_task_end_time    cur_task_start_time   cur_task_end_time
                #                                                             Max_task_end_time
                #  ------------------------------------------------task ?-------------------|
                All_task_kernel_time=All_task_kernel_time-Max_task_end_time+cur_task_start_time
            elif cur_task_start_time <= Max_task_end_time and cur_task_end_time < Max_task_end_time:
                #  |---------------task j-1-------------------|    |-----task j---------------------------|
                #  before_task_start_time   before_task_end_time    cur_task_start_time   cur_task_end_time
                #                                                                                    Max_task_end_time
                #  ------------------------------------------------task ?--------------------------------------------|
                All_task_kernel_time=All_task_kernel_time-cur_task_duration_time
        elif before_task_end_time > cur_task_start_time and before_task_end_time <= cur_task_end_time :
            if  cur_task_end_time >= Max_task_end_time:
                #  |---------------task j-1-------------------|
                #  before_task_start_time   before_task_end_time
                #                             |-----task j-------------------|
                #                           cur_task_start_time   cur_task_end_time
                #                                       Max_task_end_time
                #  ----------------------------------task ?-------------|
                All_task_kernel_time=All_task_kernel_time-before_task_end_time+cur_task_start_time
            elif cur_task_end_time < Max_task_end_time:
                #  |---------------task j-1-------------------|
                #  before_task_start_time   before_task_end_time
                #                             |-----task j-------------------|
                #                           cur_task_start_time   cur_task_end_time
                #                                                                Max_task_end_time
                #  ----------------------------------task ?-------------------------------------|
                All_task_kernel_time=All_task_kernel_time-cur_task_duration_time
        elif before_task_end_time > cur_task_start_time and before_task_end_time >= cur_task_end_time :
            #  |---------------task j-1-------------------|
            #  before_task_start_time   before_task_end_time
            #      |-----task j-------------------|
            #   cur_task_start_time   cur_task_end_time
            All_task_kernel_time=All_task_kernel_time-cur_task_duration_time

        if max(before_task_end_time,cur_task_end_time) >= Max_task_end_time:
            Max_task_end_time=max(before_task_end_time,cur_task_end_time)

    #result
    if args.type:
        print("vec_aicpu_time:",All_task_kernel_time)
    else:
        print("e2e_time:",All_task_kernel_time)
