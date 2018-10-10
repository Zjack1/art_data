#coding:utf-8
import re
import xlwt
import xlrd
import os
from xlutils.copy import copy

#rb=xlrd.open_workbook("D:/Porfile/Desktop/1.xls")####原来的excel
#wb = copy(rb)
#ws = wb.get_sheet(0)

wb = xlwt.Workbook() ####生成新的Excel
ws = wb.add_sheet('Sheet1') 
####写入测试数据的版本
b2=['16bit-old-dma','16bit-new-dma','16-','Precent']
####测试数据的网络层
b1=['conv0','conv1/dw','conv1','conv2/dw','conv2','conv3/dw','conv3','conv4/dw','conv4',
'conv5/dw','conv5','conv6/dw','conv6','conv7/dw','conv7','conv8/dw','conv8','conv9/dw',
'conv9','conv10/dw','conv10','conv11/dw','conv11',
'conv12/dw','conv12','conv13/dw','conv13','conv14_1','conv14_2','conv15_1','conv15_2',
'conv16_1','conv16_2','conv17_1','conv17_2','conv17_2_mbox_loc_','conv17_2_mbox_conf_new_',
'conv16_2_mbox_loc_','conv16_2_mbox_conf_new_','conv15_2_mbox_loc_','conv15_2_mbox_conf_new_',
'conv14_2_mbox_loc_','conv14_2_mbox_conf_new_','conv13_mbox_loc_','conv13_mbox_conf_new_',
'conv11_mbox_loc','conv11_mbox_conf_new_']
####每层需要对比的数据
b0=['TestPrepare','before forward compute','InputNumberForCircles','GroupNumberForCircles',
'InYNumberForCircles','InXNumberForCircles','OutputNumberPerGroupForCircles',
'OutTileDMAUpdate','In Tile Enqueue','In Tile DMA wait','Pad Input','Weights Enqueue',
'OutPut Store Enqueue','Kernel','Weights and Output DMA wait','total']

k2=len(b2)

k0=len(b0)

for i in range(len(b0)):
	ws.write(1,1+k2*i,b0[i])
	
for i in range(len(b1)):
	ws.write(3+i,0,b1[i])
	
a=[]
for i in range(0,k0):
	for j in range(len(b2)):
		a.append(b2[j])
#print(a)
for i in range(len(a)):
	ws.write(2,1+i,a[i])

#添加需要提取的TXT文本地址
fn = open("D:\\Porfile\\Desktop\\16bit-new-dma.txt",'rb')
fo = open("D:\\Porfile\\Desktop\\16bit-old-dma.txt",'rb')
#f1 = open("D:\\Porfile\\Desktop\\111.txt",'a')
#正则提取关键数据
findword1="Default TestPrepare.*"
findword2="Default Test0.*"
findword3="InputNumberForCircles.*"
findword4="GroupNumberForCircles.*"
findword5="InYNumberForCircles.*"
findword6="InXNumberForCircles.*"
findword7="OutputNumberPerGroupForCircles.*"
findword8="OutTileDMAUpdate.*"
findword9="Default Test2.*"
findword10="Default Test3.*"
findword11="Default Test4.*"
findword12="Default Test5.*"
findword13="Default Test1.*"
findword14="Kernel.*"
findword15="polling.*"
findword16="layer Total.*"

new_data=fn.readlines()
c_new=[]
for i in range(k0):
	all=locals()['findword'+str(i+1)]
	print('loading......'+all)
	#f1.write('------------'+str(i+1)+'-----------------\n')
	cc=[]
	for n in range(len(new_data)):
		new_line=str(new_data[n])
		if n<len(new_data)-1:
			if str(new_data[n])==str(new_data[n+1]):
				continue
		pattern=re.compile(all)
		results=pattern.findall(new_line)
		if len(results)==0:
			continue
		a=str(results)+'\n'
		a = re.sub(r'\[\"'+all+'.* cycles ', "", a)
		a=re.sub(r'\\\\r\\\\n\'\"\]','',a)
		#f1.write(a)
		cc.append(a)
	c_new.append(cc)

#print(len(c_new))	
fn.close()
#for i in range(len(c_new)):
#	for j in range(len(c_new[1])):
#		ws.write(3+j,2+i*6,int(c_new[i][j]))

old_data=fo.readlines()
d_old=[]
for i in range(k0):
	all=locals()['findword'+str(i+1)]
	print('loading......'+all)
	dd=[]
	for n in range(len(old_data)):
		old_line=str(old_data[n])
		if n<len(old_data)-1:
			if str(old_data[n])==str(old_data[n+1]):
				continue
		pattern=re.compile(all)
		results=pattern.findall(old_line)
		if len(results)==0:
			continue
		a1=str(results)+'\n'
		a1 = re.sub(r'\[\"'+all+'.* cycles ', "", a1)
		a1=re.sub(r'\\\\r\\\\n\'\"\]','',a1)
		#f1.write(a1)
		dd.append(a1)
	d_old.append(dd)

fo.close()
for i in range(len(d_old)):
	for j in range(len(d_old[1])):
		ws.write(3+j,1+i*k2,int(d_old[i][j]))####起始写入坐标
		ws.write(3+j,2+i*k2,int(c_new[i][j]))		
		ws.write(3+j,3+i*k2,int(d_old[i][j])-int(c_new[i][j]))
		ws.write(3+j,4+i*k2,str(round((int(d_old[i][j])-int(c_new[i][j]))*100/int(d_old[i][j]),4))+'%')
wb.save("D:/Porfile/Desktop/16bit.xls")		


