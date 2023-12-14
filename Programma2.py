#-*- coding: utf-8 -*-
#python3
import sys
import codecs
import nltk
import re

#funzione che tokenizza, POStagga e restituisce i 10 nomi di persona più frequenti
def AnalisiLinguistica(frasi, file):
    #lista che conterrà i 10 nomi di persona più frequenti
    ListaPersone = []
    #lista che conterrà tutti i token del testo
    tokensTOT =[]
    #lista che conterrà tutte le Named Entity del testo
    NamedEntityList = []
    #lista che conterrà i bigrammi POS del testo
    tokensPOStot = []
    #tokenizzo, POStaggo e assegno ad ogni nome proprio la sua entità
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        analisi = nltk.ne_chunk(tokensPOS)
        IOBFormat = nltk.chunk.tree2conllstr(analisi)
        for nodo in analisi:
            NE = ''
            #se l'entità è "PERSON", aggiungo il nome alla lista
            if hasattr(nodo, 'label') and nodo.label() == 'PERSON':
                for partNE in nodo.leaves():
                    NE = NE +' '+ partNE[0]
                NamedEntityList.append(NE)
        tokensTOT = tokensTOT + tokens
        tokensPOStot = tokensPOStot + tokensPOS
    #ordino i nomi di persona per frequenza e creo una ListaNomi che contenga i 10 nomi più frequenti e la loro frequenza
    Distribuzione = nltk.FreqDist(NamedEntityList)
    ListaNomi = Distribuzione.most_common(10)
    #stampo i 10 nomi più frequenti e la loro frequenza
    print("Le dieci persone più frequenti nel file", file, "sono:")
    for elem in ListaNomi:
        print("\"",elem[0].strip(),"\" con frequenza:", elem[1])
        ListaPersone.append(elem[0].strip())
    return tokensTOT, tokensPOStot, NamedEntityList, ListaPersone

#funzione che per ogni nome di persona tra i 10 più frequenti trova le frasi in cui appare
def FrasiNomi(frasi, persone):
    #lista che conterrà le frasi
    ListaFrasi = []
    #per ogni nome di persona tokenizzo il testo, se il nome appare in una frase, aggiungo la frase alla lista
    for nome in persone:
        for frase in frasi:
            tokens = nltk.word_tokenize(frase)
            if nome in frase:
                ListaFrasi.append(frase)
    #restituisco la lista con le frasi
    return ListaFrasi

#funzione che per ogni nome tra i 10 più frequenti trova la frase più lunga in cui compare
def FrasiLunghe(frasi, persone):
    #lista che conterrà le frasi più lunghe
    ListaFrasiMAX = []
    #per ogni nome di persona tra i 10 più frequenti calcolo la lunghezza delle frasi in cui compare
    for nome in persone:
        LunghMAX = 0
        Lungh = 0
        for frase in frasi:
            tokens = nltk.word_tokenize(frase)
            Lungh = len(tokens) 
            if nome in frase:
                #se la lunghezza della frase è maggiore della lunghezza massima, la lunghezza della frase diventa la limghezza massima
                if Lungh > LunghMAX:
                    LunghMAX = Lungh
                    FraseMAX = frase
        #aggiungo la frase più lunga alla lista
        ListaFrasiMAX.append(FraseMAX)
    #restituisco la lista
    return ListaFrasiMAX

#funzione che per ogni nome tra i 10 più frequenti trova la frase più breve in cui compare  
def FrasiBrevi(frasi, persone):
    #lista che conterrà le frasi più brevi
    ListaFrasiMIN = []
    #per ogni nome di persona tra i 10 più frequenti calcolo la lunghezza delle frasi in cui compare
    for nome  in persone:
        LunghMIN = 1000000000
        Lungh = 0
        for frase in frasi:
            tokens = nltk.word_tokenize(frase)
            Lungh = len(tokens)
            if nome in frase:
                #se la lunghezza della frase è minore della lunghezza minima, la lunghezza della frase diventa la limghezza minima
                if Lungh < LunghMIN:
                    LunghMIN = Lungh
                    FraseMIN = frase
        #aggiungo la frase più breve alla lista 
        ListaFrasiMIN.append(FraseMIN)
    #restituisco la lista
    return ListaFrasiMIN

#funzione che trova i 10 luoghi più frequenti nelle frasi dove compaiono i dieci nomi di persona più frequenti
def LuoghiFrequenti(LisF, persone, file):
    #lista che conterrà i token del testo
    tokensTOT = []
    #lista che conterrà i bigrammiPOS del testo
    tokensPOStot = []
    #tokenizzo, POStaggo e assegno a ogni nome proprio la sua entità per ogni nome tra i 10 più frequenti
    for nome in persone:
        #lista che conterrà i 10 luoghi più comuni
        ListaLuoghi = []
        NamedEntityList = []
        for frase in LisF:
            tokens = nltk.word_tokenize(frase)
            tokensPOS = nltk.pos_tag(tokens)
            analisi = nltk.ne_chunk(tokensPOS)
            IOBformat = nltk.chunk.tree2conllstr(analisi)
            #analizzo le frasi dove compaiono i 10 nomi di persona più frequenti
            if nome in frase:
                for nodo in analisi:
                    NE = ''
                    #se un nome è un nome di luogo lo aggiungo alla lista
                    if hasattr(nodo, 'label') and nodo.label() == "GPE":
                        for partNE in nodo.leaves():
                            NE = NE + ' ' + partNE[0]
                        NamedEntityList.append(NE)
        #ordino per frequenza i nomi di luogo
        Distribuzione = nltk.FreqDist(NamedEntityList)
        #aggiungo alla lista i 10 nomi di luogo più frequenti
        ListaLuoghi = Distribuzione.most_common(10)
        #stampo i 10 nomi di luogo più frequenti e la relativa frequenza
        print("I 10 luoghi più frequenti nelle frasi in cui è contenuto il nome \"",nome,"\" nel file", file, "sono:")
        for elem in ListaLuoghi:
            print("\"",elem[0].strip(), "\" con frequenza:", elem[1])

#funzione che trova i 10 nomi di persona più frequenti nelle frasi dove appaiono i 10 nomi di persona più frequenti
def PersoneFrequenti(frasi, persone, file):
    #lista che conterrà tutti i token del testo
    tokensTOT = []
    #lista che conterrà tutti i bigrammiPOS del testo
    tokensPOStot = []
    #tokenizzo, POStaggo e assegno a ogni nome proprio la sua entità per ogni nome tra i 10 più frequenti 
    for nome in persone:
        ListaPersone = []
        NamedEntityList = []
        for frase in frasi:
            tokens = nltk.word_tokenize(frase)
            tokensPOS = nltk.pos_tag(tokens)
            analisi = nltk.ne_chunk(tokensPOS)
            IOBformat = nltk.chunk.tree2conllstr(analisi)
            #analizzo le frasi dove compaiono i 10 nomi di persona più frequenti
            if nome in frase:
                for nodo in analisi:
                    NE = ''
                    #se un nome è un nome di luogo lo aggiungo alla lista
                    if hasattr(nodo, 'label') and nodo.label() == "PERSON":
                        for partNE in nodo.leaves():
                            NE = NE + ' ' +  partNE[0]
                        NamedEntityList.append(NE)
        #ordino per frequenza i nomi di persona
        Distribuzione = nltk.FreqDist(NamedEntityList)
        #creo una lista che contiene i 10 nomi di persona più frequenti 
        ListaPersone = Distribuzione.most_common(10)
        #stampo i 10 nomi di persona più frequenti e la relativa frequenza
        print("Le 10 persone più frequenti nelle frasi in cui è contenuto il nome \"",nome,"\" nel file", file, "sono:")
        for elem in ListaPersone:
            print("\"",elem[0].strip(), "\" con frequenza:", elem[1])

#funzione che trova i 10 sostantivi più frequenti nelle frasi in cui compaiono i 10 nomi di persona più frequenti
def SOSFrequenti(Lis, persone, file):
    #lista che conterrà i bigrammiPOS
    tokensPOStot = []
    #lista che conterrà i token 
    tokensTOT = []
    #tokenizzo e POStaggo per ognuno dei 10 nomi di persona più frequenti nel testo
    for nome in persone:
        listaSOS = []
        for frase in Lis:
            tokens = nltk.word_tokenize(frase)
            tokensPOS = nltk.pos_tag(tokens)
            #analizzo le frasi che contengono i dieci nomi di persona più frequenti nel testo
            if nome in frase:
                #se il token è  un sostantivo lo aggiungo alla lista
                for bigramma in tokensPOS:
                    if bigramma[1] in ["NN", "NNP", "NNS", "NNPS"]:
                        listaSOS.append(bigramma[0])
        #ordino per frequenza i sostantivi
        Distribuzione = nltk.FreqDist(listaSOS)
        #creo una lista che contiene i 10 sostantivi più frequenti
        ListaSostantivi =  Distribuzione.most_common(10)
        #stampo i 10 sostantivi più frequenti e la relativa frequenza
        print("I 10 sostantivi più frequenti nelle frasi in cui è contenuto il nome \"",nome,"\" nel file", file, "sono:")
        for elem in ListaSostantivi:
            print("\"",elem[0], "\" con frequenza:", elem[1])

#funzione che trova i 10 verbi più frequenti nelle frasi in cui compaiono i 10 nomi di persona più frequenti 
def VBFrequenti(Lis, persone, file):
    #tokenizzo e POStaggo
    for nome in persone:
        listaVB = []
        for frase in Lis:
            tokens = nltk.word_tokenize(frase)
            tokensPOS = nltk.pos_tag(tokens)
            #analizzo le frasi in cui compaiono i 10 nomi di persona più frequenti nel testo
            if nome in frase:
                #se il token è un verbo lo aggiungo alla lista
                for bigramma in tokensPOS:
                    if(bigramma[1] in ["VB","VBZ","VBD","VBG","VBN","VBP"]):
                        listaVB.append(bigramma[0])
        #ordino i verbi per frequenza
        Distribuzione = nltk.FreqDist(listaVB)
        #creo una lista che contenga i 10 verbi più frequenti
        ListaVerbi = Distribuzione.most_common(10)
        #stampo i 10 verbi più frequenti e la relativa frequenza
        print("I 10 verbi più frequenti nelle frasi contenenti in cui è contenuto il nome \"",nome,"\" nel file", file, "sono:")
        for elem in ListaVerbi:
            print("\"",elem[0], "\" con frequenza:", elem[1])

#funzione che calcola la probabilità di una frase secondo un modello di Markov di ordine zero
def CalcolaProbabilitaFraseMarkovZero(LunghezzaCorpus, DistribuzioneDiFrequenza, frase):
    probabilita = 1.0
    for tok in frase:
        #la probabilità di un token è data dalla sua frequenza fratto la lunghezza del corpus
        probabilitaToken = (DistribuzioneDiFrequenza[tok]*1.0/LunghezzaCorpus*1.0)
        probabilita = probabilita*probabilitaToken
    return probabilita

#funzione che per ogni per ogni nome di persona tra i 10 più frequenti stampa la frase formata da almeno 8 token e al massimo 12 token con la probabilità più elevata, calcolata con un modello di Markov di ordine zero
def Markov(LiSTAf, TestoToken, persone, file):
    #calcolo la lunghezza del testo
    LunghezzaCorpus = len(TestoToken)
    #trovo la frequenza di ogni token
    DistribuzioneDiFrequenzaToken = nltk.FreqDist(TestoToken)
    for nome in persone:
        probabilitaMAX = 0.0
        for frase in LiSTAf:
            tokensFrase = nltk.word_tokenize(frase)
            #analizzo le frasi in cui compaiono i nomi di persona più frequenti nel testo
            if nome in frase:
                #se la frase è lunga almeno 8 token e al massimo 12, calcolo la probabilità della frase
                if len(tokensFrase) >= 8 and len(tokensFrase) <= 12:
                    Probabilita = CalcolaProbabilitaFraseMarkovZero(LunghezzaCorpus, DistribuzioneDiFrequenzaToken, tokensFrase)
                    #se la probabilità della frase è maggiore della probabilità massima, la probabilità della frase diventa la probabilità massima
                    if Probabilita > probabilitaMAX:
                        probabilitaMAX = Probabilita
                        FraseMAX = frase
        #stampo la frase e la relativa probabilità
        print("La frase con probabilità massima che contiene il nome \"",nome, "\" nel file", file , "è:", FraseMAX, "con probabilità:", probabilitaMAX)
        
#funzione che per ogni nome di persona tra i 10 più frequenti trova date, mesi e giorni della settimana 
def DateMesiGiorni(LisT, persone, file):
    for nome in persone:
        #lista che conterrà i giorni, i mesi e le date
        ListaGiorniMesi = []
        for frase in LisT:
            tokens = nltk.word_tokenize(frase)
            #analizzo le frasi in cui compaiono i 10 nomi di persona più frequenti nel testo
            if nome in frase:
                ListaMatch = re.findall(r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\d*\sJanuary\s\d*|\d*\sFebruary\s\d*|\d*\sMarch\s\d*|\d*\sApril\s\d*|\d*\sMay\s\d*|\d*\sJune\s\d*|\d*\sJuly\s\d*|\d*\sAugust\s\d*|\d*\sSeptember\s\d*|\d*\sOctober\s\d*|\d*\sNovember\s\d*|\d*\sDecember\s\d*', frase)
                if not ListaMatch == []:
                    ListaGiorniMesi = ListaGiorniMesi + ListaMatch
                    #ordino la lista per frequenza
                    ListaDist = nltk.FreqDist(ListaGiorniMesi)
                    ListaGiorniMesiAnni = ListaDist.most_common(len(ListaDist))
        #stampo i giorni, mesi e date più frequenti con la relativa frequenza
        print("I giorni, mesi e date più frequenti nelle frasi in cui è contenuto il nome \"",nome,"\" nel file", file, "sono:")
        for elem in ListaGiorniMesiAnni:
            print("\"",elem[0],"\" con frequenza:", elem[1])
    
def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #divido i file in frasi
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    #invoco tutte le funzioni
    TestoTokenizzato1, TestoAnalizzatoPOS1, NamedEntityList1, ListaPeople1 = AnalisiLinguistica(frasi1, file1)
    TestoTokenizzato2, TestoAnalizzatoPOS2, NamedEntityList2, ListaPeople2 = AnalisiLinguistica(frasi2, file2)
    ListaF1 = FrasiNomi(frasi1, ListaPeople1)
    ListaF2 = FrasiNomi(frasi2, ListaPeople2)
    print("Le frasi contenenti le  10 persone più frequenti nel file", file1, "sono:", ListaF1)
    print("Le frasi contenenti le 10 persone più frequenti nel file", file2, "sono:", ListaF2)
    print("Le frasi più lunghe contenenti i 10 nomi di persona più frequenti nel file", file1, "sono:", FrasiLunghe(ListaF1, ListaPeople1))
    print("Le frasi più lunghe contenenti i 10 nomi di persona più frequenti nel file", file2, "sono:", FrasiLunghe(ListaF2, ListaPeople2))
    print("Le frasi più brevi contenenti i 10 nomi di persona più frequenti nel file", file1,"sono:", FrasiBrevi(ListaF1, ListaPeople1))
    print("Le frasi più brevi contenenti i 10 nomi di persona più frequenti nel file", file2, "sono:", FrasiBrevi(ListaF2, ListaPeople2))
    LuoghiFrequenti(ListaF1, ListaPeople1, file1)
    LuoghiFrequenti(ListaF2, ListaPeople2, file2)
    PersoneFrequenti(ListaF1, ListaPeople1, file1)
    PersoneFrequenti(ListaF2, ListaPeople2, file2)
    SOSFrequenti(ListaF1, ListaPeople1, file1)
    SOSFrequenti(ListaF2, ListaPeople2, file2)
    VBFrequenti(ListaF1, ListaPeople1, file1)
    VBFrequenti(ListaF2, ListaPeople2, file2)
    DateMesiGiorni(frasi1, ListaPeople1, file1)
    DateMesiGiorni(frasi2, ListaPeople2, file2)
    LunghezzaCorpus1 = len(TestoTokenizzato1)
    LunghezzaCorpus2 = len(TestoTokenizzato2)
    DistribuzioneDiFrequenzaToken1 = nltk.FreqDist(TestoTokenizzato1)
    DistribuzioneDiFrequenzaToken2 = nltk.FreqDist(TestoTokenizzato2)
    CalcolaProbabilitaFraseMarkovZero(LunghezzaCorpus1, DistribuzioneDiFrequenzaToken1, frasi1)
    CalcolaProbabilitaFraseMarkovZero(LunghezzaCorpus2, DistribuzioneDiFrequenzaToken2, frasi2)
    Markov(frasi1, TestoTokenizzato1, ListaPeople1, file1)
    Markov(frasi2, TestoTokenizzato2, ListaPeople2, file2)

main(sys.argv[1], sys.argv[2])

