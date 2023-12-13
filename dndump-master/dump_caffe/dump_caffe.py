import sys
import os
import caffe
import pdb
import xlsxwriter
import datetime
import time
import sys
import argparse

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototxt", help="prototxt path")
    parser.add_argument("--optype", default="NULL",  help="search bottom and top optype of optype from Network")
    args = parser.parse_args()

    return args

args = get_args()
print(args)

model_prototxt=args.prototxt
optype=args.optype


input_shape=''
output_shape=''
filter_shape=''
rank_op={}
unrank_op={}
temp_op=None
multi_op = {}
all_op = {}

COL={
    'Layer':'A',
    'Type':'B',
    'bottom':'C',
    'top':'D',
    'input_shape':'E',
    'output_shape':'F',
    'filter_shape':'G',
    'bias_shape':'H'
}


#[Begin]find the top op & bottom op & top_and_bottom op of optype
def find_optype_by_topname_excep(net,topname):
    for i in range(len(net.top_vecs)):
        if(net.bottom_names[net._layer_names[i]].count(topname)==1):
            result= net.layers[i].type 
            return result
    return None
    
def find_optye_by_name(k,net,layer_names):
    result = []
    for  name in layer_names:
        tmp=None
        for i in range(len(net.top_vecs)):
            if(name==net._layer_names[i] and i!=k):
                tmp= net.layers[i].type 
            elif(name==net._layer_names[i] and i==k):
                tmp=find_optype_by_topname_excep(net,name)
        if(tmp is None):
            tmp=find_optype_by_topname_excep(net,name)
        if(tmp is not None):
            result.append(tmp)
    return result

def find_top_op(i,net,optype):
    global multi_op
    if(optype == net.layers[i].type):
        res=str([optype])+"--->"+str(find_optye_by_name(i,net,net.top_names[net._layer_names[i]]))
        if(multi_op.get(res) is not None ):
            multi_op[res]+=1
        else:
            multi_op.update({res:1})

def find_bottom_op(i,net,optype):
    global multi_op
    if(optype == net.layers[i].type):
        res=str(find_optye_by_name(i,net,net.bottom_names[net._layer_names[i]])) + "--->"+ str([optype]) 
        if(multi_op.get(res) is not None ):
            multi_op[res]+=1
        else:
            multi_op.update({res:1})

def find_top_and_bottom_op(i,net,optype):
    global multi_op
    if(optype == net.layers[i].type):
        res=str(find_optye_by_name(i,net,net.bottom_names[net._layer_names[i]])) + "--->"+ str([optype]) + "--->"+str(find_optye_by_name(i,net,net.top_names[net._layer_names[i]])) 
        if(multi_op.get(res) is not None ):
            multi_op[res]+=1
        else:
            multi_op.update({res:1})
#[End]find the top op & bottom op & top_and_bottom op of optype

#[Begin]find the rank & unrank op of net
def add_dict(result):
    optype=result.split('|')[1]
    ishape=result.split('|')[4]
    oshape=result.split('|')[5]
    global temp_op
    global rank_op
    global unrank_op

    if ishape == oshape:
        if optype in unrank_op:
            unrank_op[optype]+=1
            if temp_op is not None:
                if temp_op in rank_op:
                    rank_op[temp_op]+=1
                else:
                    tmp_dict={temp_op:1}
                    rank_op.update(tmp_dict)
                temp_op=None
        else:
            if temp_op is None:
                temp_op=optype
            else:
                temp_op=temp_op + '+' + optype
        
    else:
        if optype in rank_op:
            del rank_op[optype]

        #unchange dims op type
        if temp_op is not None:
            if temp_op in rank_op:
                rank_op[temp_op]+=1
            else:
                tmp_dict={temp_op:1}
                rank_op.update(tmp_dict)
            temp_op=None
        
        #change dims op type
        if optype in unrank_op:
            unrank_op[optype]+=1
        else:
            tmp_dict={optype:1}
            unrank_op.update(tmp_dict)
#[End]find the rank & unrank op of net


netname=model_prototxt.split('/')[-1][0:20]
workbook = xlsxwriter.Workbook(netname+'.xlsx')
caffe.set_mode_cpu()
net = caffe.Net(model_prototxt,caffe.TEST,0)
worksheet = workbook.add_worksheet(netname)     
worksheet_multi = workbook.add_worksheet(netname+"_multi")
       
for key in list(COL.keys()):
    row=COL[key]+'1'
    data=[key]
    worksheet.write_row(row, data)

COL_multi={
    'multi_op':'A',
    'count':'B'
}
for key in list(COL_multi.keys()):
    row=COL_multi[key]+'1'
    data=[key]
    worksheet_multi.write_row(row, data)



for i in range(len(net.top_vecs)):
  
        col='A'
        row=col+str(i+2)        
        data = [net._layer_names[i]]
        worksheet.write_row(row, data)

        col=chr(ord(col)+1)
        row=col+str(i+2)
        data = [net.layers[i].type]
        worksheet.write_row(row, data)

        col=chr(ord(col)+1)
        row=col+str(i+2)
        data = [str(net.bottom_names[net._layer_names[i]])]
        worksheet.write_row(row, data)


        col=chr(ord(col)+1)
        row=col+str(i+2)
        data = [str(net.top_names[net._layer_names[i]])]
        worksheet.write_row(row, data)


        col=chr(ord(col)+1)
        row=col+str(i+2)
        input_shape=[net.bottom_vecs[i][j].data.shape for j in range(len(net.bottom_vecs[i]))]
        data = [str(input_shape)]
        worksheet.write_row(row, data)

        col=chr(ord(col)+1)
        row=col+str(i+2)
        output_shape=[net.top_vecs[i][j].data.shape for j in range(len(net.top_vecs[i]))]
        data = [str(output_shape)]
        worksheet.write_row(row, data)

        
        result = net._layer_names[i]
        result += '|' + net.layers[i].type 
        result += '|' + str(net.top_names[net._layer_names[i]])
        result += '|' + str(net.bottom_names[net._layer_names[i]])
        result += '|'+ str(input_shape) 
        result += '|' +str(output_shape)

        for k in range(len(net.layers[i].blobs)):
            col=chr(ord(col)+1)
            row=col+str(i+2)
            data = [str(net.layers[i].blobs[k].data.shape)]
            worksheet.write_row(row, data)
            result += '|'+ str(net.layers[i].blobs[k].data.shape)
        #pdb.set_trace()
        print("Origin---------"+result)
        #find all_op
        if(all_op.get(net.layers[i].type) is not None ):
            all_op[net.layers[i].type]+=1
        else:
            all_op.update({net.layers[i].type:1})

        #find multi_op
        if(optype!="NULL" and optype!="ALL"):
            find_top_op(i,net,optype)
            find_bottom_op(i,net,optype)
            find_top_and_bottom_op(i,net,optype)
        #find rank_op unrank_op
        add_dict(result)

#last unchange dims op type
if temp_op is not None:
    if temp_op in rank_op:
        rank_op[temp_op]+=1
    else:
        tmp_dict={temp_op:1}
        rank_op.update(tmp_dict)
    temp_op=None


if(optype=="ALL"):
    for key in all_op.keys():
        for i in range(len(net.top_vecs)):
            find_top_op(i,net,key)
            find_bottom_op(i,net,key)
            find_top_and_bottom_op(i,net,key)
    tmp=1
    for key in multi_op.keys():
        tmp+=1
        #multi_op
        col='A'
        row=col+str(tmp)
        data = [key]
        worksheet_multi.write_row(row, data)
        #count
        col=chr(ord(col)+1)
        row=col+str(tmp)
        data = [str(multi_op[key])]
        worksheet_multi.write_row(row, data)


#print result: all_op multi_op rank_op unrank_op
print('\n'+netname+' : '+str(len(net._layer_names)))
print('----------------all_op------------------------')
for i in all_op.keys():
    print(i+' : '+str(all_op[i]))
print('\n')

print('----------------multi_op('+optype+')-----------------------')
for i in multi_op.keys():
    print(i+' : '+str(multi_op[i]))
print('\n')

print('----------------rank_op--------------------------')
for i in rank_op.keys():
    print(i+' : '+str(rank_op[i]))
print('----------------unrank_op------------------------')
for i in unrank_op.keys():
    print(i+' : '+str(unrank_op[i]))
print('\n')




workbook.close()


