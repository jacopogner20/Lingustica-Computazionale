#-*- coding: utf-8 -*-                                                        
#python3
import sys
import codecs
import nltk
from nltk import bigrams
import math 

#funzione che calcola il numero delle frasi
def CalcolaNumeroFrasi(frasi):
    #lista che contiene tutte le frasi del testo
    NumeroF = []
    #tramite un ciclo for appendo tutte le frasi alla lista
    for frase in frasi:
        NumeroF.append(frase)
    #restituisco il risultato (la lunghezza della lista è uguale al numero delle frasi nel testo)
    return len(NumeroF)

#funzione che calcola il numero di token in un testo
def CalcolaLunghezza(frasi):
    #variabile usata per la lunghezza del testo
    lunghezzaTOT = 0
    for frase in frasi:
        #divido la frase in token
        tokens = nltk.word_tokenize(frase)
        #calcolo la lunghezza sommando il valore della variabile e la lunghezza di ogni token                                      
        lunghezzaTOT = lunghezzaTOT + len(tokens)
    #restituisco il risultato (il valore della variabile è uguale al numero di tokens nel testo)                             
    return lunghezzaTOT

#funzione che calcola il numero di caratteri in un testo
def CalcolaCaratteri(frasi):
    #variabile usata per la lunghezza dei token nel testo
    caratteri = 0
    for frase in frasi:
        #divido la frase in token
        tokens = nltk.word_tokenize(frase)
        #sommo la lunghezza di ogni token alla variabile
        for tok in tokens:
            caratteri = caratteri + len(tok)
    #restituisco il risultato (il valore della variabile è uguale al numero di caratteri nel testo)
    return caratteri
        
#funzione che calcola la grandezza di un vocabolario in un testo
def CalcolaVoc(frasi):
    #variabile usata per la lunghezza del vocabolario del testo
    lunghezzaVoc = 0.0
    #lista che contiene tutti i tokens del testo
    tokensTOT = []
    #tokenizzo il testo e aggiungo ogni token alla lista "tokensTOT"
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensTOT = tokensTOT + tokens
    #il valore della variabile è uguale alla lunghezza della della lista dei token senza alcuna ripetizione (ottengo ciò tramite "set")
    lunghezzaVoc = len(set(tokensTOT))
    #restituisco il risultato
    return lunghezzaVoc

#funzione che POS-tagga
def AnnotazioneL(frasi):
    #lista che conterrà il testo POStaggato
    tokensPOStot = []
    #tokenizzo il testo e lo POStaggo 
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensPOStot = tokensPOStot + tokensPOS
    #restituisco la lista che contiene il testo POStaggato
    return tokensPOStot

#funzione che crea la lista di tutte le POS del testo
def EstraiSequenzaPOS(TestoAnPOS):
    #lista che conterrà tutte le POS del testo
    listaPOS = []
    #aggiungo alla lista tutte le POS del testo
    for bigramma in TestoAnPOS:
        listaPOS.append(bigramma[1])
    #restituire il risultato
    return listaPOS

#funzione che calcola il rapporto tra sostantivi e verbi 
def SOSVB(TestoPOS):
    #variabile che tiene il conto di tutti i sostantivi all'interno del testo
    Sostantivi = 0.0
    #variabile che tiene il conto di tutti i verbi all'interno del testo
    Verbi = 0.0
    for bigramma in TestoPOS:
        #se la parola è un sostantivo, aggiungo 1 alla variabile Sostantivi
        if(bigramma[1] in ["NNP","NN","NNS","NNPS"]):
            Sostantivi = Sostantivi + 1
        #se la parola è un verbo, aggiungo 1 alla variabile Verbi
        elif(bigramma[1] in ["VB","VBD","VBG","VBN","VBP","VBZ"]):
            Verbi = Verbi + 1
    #variabile che calcola il rapporto tra il numero di sostantivi e il numero di verbi
    rapporto = Sostantivi/Verbi
    #restituisco il risultato
    return rapporto

#funzione che stampa le dieci PoS più frequenti
def POSFrequenti(TestoPOS, file):
    #lista che conterrà tutte le POS del testo
    listaPOS = []
    #aggiungo alla lista tutte le POS del testo
    for bigramma in TestoPOS:
        listaPOS.append(bigramma[1])
    print("Le 10 POS più frequenti nel file", file, "sono:")
    for pos in set(listaPOS):
        #ordino per frequenza in ordine decrescente tutte le POS in listaPOS
        distribuzione = nltk.FreqDist(listaPOS)
        #prendo le 10 POS più frequenti
        ListaDieci = distribuzione.most_common(10)
    #stampo ogni POS con la propria frequenza
    for elem in ListaDieci:
        print(elem[0], "con frequenza:",  elem[1])

#funzione che trova  i bigrammiPOS con probabilità condizionata massima
def CercaBigrammaProbCondMAX(TestoAnPOS, bigrammi, bigDiv):
    #lista che conterrà i bigrammi e le probabilità condizionate
    ListaSuperProbMAX = []
    probCondizionataMAX = 0.0
    #per ogni bigramma calcolo la probabilità condizonata
    for bigramma in bigDiv:
        freqBig = bigrammi.count(bigramma)
        freqA = TestoAnPOS.count(bigramma[0])
        probCond = freqBig*1.0/freqA*1.0
        #se la probabilità condizionata del bigramma è maggiore della proabilità condizionata massima, la probabilità condizonata del bigramma diventa la nuova probabilità condizonta massima 
        if probCond >= probCondizionataMAX:
            probCondizionataMAX = probCond
            bigrammaMAX = bigramma
            #aggiungo alla lista i bigrammi con probabilità condizionata massima e la loro probabilità condizionata
            ListaSuperProbMAX.append(bigrammaMAX)
            ListaSuperProbMAX.append(probCondizionataMAX)
    #restituisco la lista
    return ListaSuperProbMAX

#funzione che trova i bigrammiPOS con forza associativa massima
def CercaBigrammaForzaAssMAX(TestoPOS, bigrammi):
    #lista che conterrà i bigrammi e le forze di associazione
    ListaSuperForza = []
    ForzaAssMAX = -99999999999.0
    #per ogni bigramma calcolo la forza associativa 
    for bigramma in set(bigrammi):
        freqBigs = bigrammi.count(bigramma)
        freqAs = TestoPOS.count(bigramma[0])
        freqCs = TestoPOS.count(bigramma[1])
        MI = (freqBigs/len(TestoPOS))/((freqAs/len(TestoPOS))*(freqCs/len(TestoPOS)))
        ForzaAssociativa = math.log(MI, 2)
        #se la forza associativa del bigramma è maggiore della forza associativa massima, la forza associativa del bigramma diventa la nuova forza associativa massima
        if ForzaAssociativa >= ForzaAssMAX:
            ForzaAssMAX = ForzaAssociativa
            bigrammaForza = bigramma
            #aggiungo alla lista i bigrammi con forza associativa massima e la loro forza associativa
            ListaSuperForza.append(bigrammaForza)
            ListaSuperForza.append(ForzaAssMAX)
    #restituisco il risultato
    return ListaSuperForza

#funzione che calcola e stampa la distribuzione degli hapax ogni mille token
def CalcolaHapax(frasi, file):
    #variabile usata per la lunghezza del testo
    lungh = 0.0
    #lista che conterrà tutti i token del testo
    tknsT = []
    #tokenizzo il testo e aggiungo tutti i token del testo alla lista
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tknsT = tknsT + tokens
    #calcolo la lunghezza del testo
    lungh = len(tknsT)
    #variabile contatore 
    contatore = 1000
    #ogni mille token calcolo quanti hapax ci sono
    while(contatore <= lungh):
        MilleHapax = nltk.FreqDist(tknsT[0:contatore])
        #variabile usata per contare quanti hapax ci sono
        Hapax = 0
        #se il token ha frequenza 1 aggiungo 1 alla variabile che conta il numero degli hapax
        for tok in MilleHapax:
            if MilleHapax[tok] == 1:
                Hapax = Hapax + 1
        #aggiorno il contatore
        contatore = contatore + 1000
        #stampo la distribuzione degli hapax ogni mille token e il numero degli hapax ogni mille token
        print("Nei primi", contatore - 1000, "token del file",file," la distribuzione degli hapax è",Hapax/(contatore - 1000),"e ci sono", Hapax, "hapax")

#funzione che stampa i bigrammiPOS con probabilità condizionata massima e con forza associativa massima in ogni testo 
def StampaBigrammiPOS(probcond, forzass, file):
    n = 0
    print("I bigrammi POS con probabilità condizionata massima nel file", file, "sono:")
    while n < len(probcond):
        print(probcond[n], probcond[n + 1])
        n = n + 2
    c = 0
    print("I bigrammi POS con forza associativa massima nel file", file, "sono:")
    while c < len(forzass):
        print(forzass[c], forzass[c + 1])
        c = c + 2

def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #divido i file in frasi     
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    #per ogni funzione creo una variabile che contegga il risultato
    NumeroFrasi1 = CalcolaNumeroFrasi(frasi1)
    NumeroFrasi2 = CalcolaNumeroFrasi(frasi2)
    lunghezza1 = CalcolaLunghezza(frasi1)
    lunghezza2 = CalcolaLunghezza(frasi2)
    Rapporto1 = CalcolaLunghezza(frasi1)/CalcolaNumeroFrasi(frasi1)
    Rapporto2 = CalcolaLunghezza(frasi2)/CalcolaNumeroFrasi(frasi2)
    RapCar1 = CalcolaCaratteri(frasi1)/CalcolaLunghezza(frasi1)
    RapCar2 = CalcolaCaratteri(frasi2)/CalcolaLunghezza(frasi2)
    Voc1 = CalcolaVoc(frasi1)
    Voc2 = CalcolaVoc(frasi2)
    TestoAnalizzatoPOS = AnnotazioneL(frasi1)
    TestoAnalizzatoPOS2 = AnnotazioneL(frasi2)
    SequenzaPOS1 = EstraiSequenzaPOS(TestoAnalizzatoPOS)
    SequenzaPOS2 = EstraiSequenzaPOS(TestoAnalizzatoPOS2)
    SV = SOSVB(TestoAnalizzatoPOS)
    SV2 = SOSVB(TestoAnalizzatoPOS2)
    bigrammi1 = list(bigrams(SequenzaPOS1))
    bigrammi2 = list(bigrams(SequenzaPOS2))
    bigrammiDiversi1 = set(bigrammi1)
    bigrammiDiversi2 = set(bigrammi2)
    ProbCond1  = CercaBigrammaProbCondMAX(SequenzaPOS1, bigrammi1, bigrammiDiversi1)
    ProbCond2 = CercaBigrammaProbCondMAX(SequenzaPOS2, bigrammi2, bigrammiDiversi2)
    ForzAss1 = CercaBigrammaForzaAssMAX(SequenzaPOS1, bigrammi1)
    ForzAss2 = CercaBigrammaForzaAssMAX(SequenzaPOS2, bigrammi2)
    #stampo i risultati
    print("Il file", file1, "è lungo", lunghezza1, "token")
    print("Il file", file2, "è lungo", lunghezza2, "token")
    if(lunghezza1 > lunghezza2):
        print("Il file", file1,  "è più lungo del file", file2)
    elif(lunghezza1 < lunghezza2):
        print("Il file", file2, "è più lungo del file", file1)
    else:
        print("I due file hanno lo stesso numero di token")
    print("Il file", file1, "è composto da", NumeroFrasi1, "frasi")
    print("Il file", file2, "è composto da", NumeroFrasi2, "frasi")
    if NumeroFrasi1 > NumeroFrasi2:
        print("Il file", file1, "è composto da più frasi del file", file2)
    elif NumeroFrasi1 < NumeroFrasi2:
        print("Il file", file2, "è composto da più frasi del file", file1)
    else:
        print("I due file sono composti dallo stesso numero di frasi")
    print("Nel file", file1, "ci sono in media", Rapporto1, "tokens per ogni frase")
    print("Nel file", file2, "ci sono in media", Rapporto2, "tokens per ogni frase")
    print("Nel file", file1, "ci sono in media", RapCar1, "caratteri per token")
    print("Nel file", file2, "ci sono in media", RapCar2, "caratteri per token")
    print("Il vocabolario del file", file1, "è lungo", Voc1, "token")
    print("Il vocabolario del file", file2, "è lungo", Voc2, "token")
    if(Voc1 > Voc2):
        print("Il vocabolario del file", file1, "è più esteso del vocabolario del file", file2)
    elif(Voc2 > Voc1):
        print("Il vocabolario del file", file2, "è più esteso del vocabolario del file", file1)
    else:
        print("I vocabolari dei due file hanno la stessa estensione")
    print("Nel file", file1, "il rapporto tra sostantivi e verbi è", SV)
    print("Nel file", file2, "il rapporto tra sostantivi e verbi è", SV2)
    POSFrequenti(TestoAnalizzatoPOS, file1)
    POSFrequenti(TestoAnalizzatoPOS2, file2)
    CalcolaHapax(frasi1, file1)
    CalcolaHapax(frasi2, file2)
    StampaBigrammiPOS(ProbCond1, ForzAss1, file1)
    StampaBigrammiPOS(ProbCond2, ForzAss2, file2)
    
main(sys.argv[1], sys.argv[2])
