from pacsltk import perfmodel
import numpy as np
import pdb    
from openpyxl import Workbook
import xlsxwriter
import os

def get_avg_res_time(arrival_rate, warm_service_time, cold_service_time,idle_time_before_kill):
        perf = perfmodel.get_sls_warm_count_dist(arrival_rate,
                                                 warm_service_time, 
                                                 cold_service_time, 
                                                 idle_time_before_kill)
        return perf[0]['avg_resp_time']

def insert_data(listdata):
	wb = xlsxwriter.Workbook("Excel1.xlsx")
	ws = wb.add_worksheet()
	row = 0
	col = 0
	for line in listdata:
		for item in line:
			ws.write(row, col, item)
			col += 1
		row += 1
		col = 0
 
	wb.close()

def main():
    np.random.seed(2)
    N=10000
    List=[]
    List.append(["Lambda", "warm_service_time", "cold_service_time","expiration_time","ave_response_time"])
    for i in range(N):
        Lambda=np.random.uniform(0.1, 0.7)
        warm_service_time=np.random.uniform(0.2, 4)
        cold1=warm_service_time+warm_service_time*(1/6)
        cold2=warm_service_time+warm_service_time*(1/4)
        cold_service_time=np.random.uniform(cold1,cold2)
        idle_time_before_kill=np.random.randint(600, 1200)
        ave_response_time=get_avg_res_time(Lambda, warm_service_time, cold_service_time,idle_time_before_kill)
        List.append([Lambda, warm_service_time, cold_service_time,idle_time_before_kill,ave_response_time])
   
    insert_data(List)
    os.system("Excel1.xlsx")


if __name__ == '__main__':
    main()