import pandas as pd
import logging

def Count(table, level):
    
    count = pd.eval('table.groupby([level]).count()')
    name = count.columns[0]
    count = count[count[name]!=0].copy()
    count = count.drop(count.columns[1:], axis=1)
    count.columns = pd.eval('[\'Quantity\']')
    count = count.sort_values(['Quantity'], ascending=False)
    count['Percent'] = pd.eval('count[\'Quantity\'].div(count[\'Quantity\'].sum())*100')
    
    print(level+": "+ str(pd.eval('count.shape[0]')))
    return pd.eval('count')

def SelectQuant_Number(sinal, quantity, countLevel, taxonomyAux, level):
    
    if(sinal=="="):
        print("Quantity equal", quantity)
        logging.info("\nQuantity equal %s\n", quantity)
        count= countLevel[countLevel['Quantity']==quantity]
        countNo = countLevel[countLevel['Quantity']!=quantity]
    elif(sinal==">"):
        print("Quantity more than", quantity)
        logging.info("\nQuantity more than %s\n", quantity)
        count= countLevel[countLevel['Quantity']>quantity]
        countNo = countLevel[countLevel['Quantity']<=quantity]
    elif(sinal=="<"):
        print("Quantity less than", quantity)
        logging.info("\nQuantity less than %s\n", quantity)
        count= countLevel[countLevel['Quantity']<quantity]
        countNo = countLevel[countLevel['Quantity']>=quantity]
    elif(sinal==">="):
        print("Quantity equal and more than", quantity)
        logging.info("\nQuantity equal and more than %s\n", quantity)
        count= countLevel[countLevel['Quantity']>=quantity]
        countNo = countLevel[countLevel['Quantity']<quantity]
    elif(sinal=="<="):
        print("Quantity equal and less than", quantity) 
        logging.info("\nQuantity equal and less than %s\n", quantity)
        count= countLevel[countLevel['Quantity']<=quantity]
        countNo = countLevel[countLevel['Quantity']>quantity]
              
    print("Labels:",count.shape[0],"\n")
    logging.info("\nLabels %s\n", count.shape[0])
    taxonomyFinal = pd.DataFrame()
    
    for i in count.index:
        #print("Unlabel: ",i)
        aux = taxonomyAux[taxonomyAux[level]==i]
        taxonomyFinal = taxonomyFinal.append(aux)
    
    return taxonomyFinal, count, countNo 


def SelectQuant_String(sinal, quantity, countLevel, taxonomyAux, level):
    
    if(sinal=="="):
        print("Quantity equal", quantity)
        logging.info("\nQuantity equal %s\n", quantity)
        count= countLevel[countLevel['Quantity']==quantity]
        countNo = countLevel[countLevel['Quantity']!=quantity]
    elif(sinal==">"):
        print("Quantity more than", quantity)
        logging.info("\nQuantity more than %s\n", quantity)
        count= countLevel[countLevel['Quantity']>quantity]
        countNo = countLevel[countLevel['Quantity']<=quantity]
    elif(sinal=="<"):
        print("Quantity less than", quantity)
        logging.info("\nQuantity less than %s\n", quantity)
        count= countLevel[countLevel['Quantity']<quantity]
        countNo = countLevel[countLevel['Quantity']>=quantity]
    elif(sinal==">="):
        print("Quantity equal and more than", quantity)
        logging.info("\nQuantity equal and more than %s\n", quantity)
        count= countLevel[countLevel['Quantity']>=quantity]
        countNo = countLevel[countLevel['Quantity']<quantity]
    elif(sinal=="<="):
        print("Quantity equal and less than", quantity) 
        logging.info("\nQuantity equal and less than %s\n", quantity)
        count= countLevel[countLevel['Quantity']<=quantity]
        countNo = countLevel[countLevel['Quantity']>quantity]
              
    print("Labels:",count.shape[0],"\n")
    logging.info("\nLabels %s\n", count.shape[0])
    taxonomyFinal = pd.DataFrame()
    
    for i in count.index:
        #print("Unlabel: ",i)
        aux = taxonomyAux[taxonomyAux[level].str.contains(i)]
        taxonomyFinal = taxonomyFinal.append(aux)
    
    return taxonomyFinal, count, countNo 