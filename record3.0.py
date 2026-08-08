import serial
import numpy as np #operazioni numeriche sugli array
import time #per la gestione del tempo
import os #per le operazioni sul sistema operativo
import pandas as pd #analisi dati strutturati(CVS)
from scipy.signal import butter, lfilter, welch #butter :calcolare i coefficienti del filtro (b e a numeratore e demonimatore)
                                                #lfilter: applicare il filtro
                                                #welch: per calcolare òa densità spettrale di potenza PSD
#====config. costanti======
SERIAL_PORT= 'COM4'
FS= 250
DURATA= 2
BAUD_RATE= 115200 #velocità di trasmissione dati seriale
OUTPUT_FILE= r'C:\User\Morena\Documents\VScode\venv\BCI_Project\dataset_eeg_3.csv'

#====filtro butterworth (1-40hz)=====
def butter_bandpass_filter (data, lowcut, highcut, fs, order=4):
    nyquist=fs/2   
    low= lowcut/nyquist #normalizziamo la fre. più bassa perchè la funzione butter prende in imput valori adimenzionali
    high= highcut/nyquist
    b,a=butter (order, [low, high], btype='band') #definiamo i coefficienti del filtro e il timpo di filtro 
    return lfilter(b, a, data) #funzione che applica il filtro
#====feature extraction======
def extract_features(eeg, fs):
    filtered= butter_bandpass_filter (eeg, 1, 40, fs, order=4) #assegno valori specifici al filtro e li savo nella variabile filtered
    freqs, psd= welch(filtered, fs, nperseg=fs*2) #con la funzione welch ricavo la frequenza e la power spectral density, nperseg(lunghezza del segmento)
                                                    #risultato: 2 array di stessa lunghezza (freq e psd) che coprono tutte le frequenze ma a me interessano solo le bande quindi definisco un indice idx per estrarre le features
                                                    #idx deve identificare quali posizioni dell'array freq contengono i valori tra 8 e 12 per la banda alpha ..ecc...
                                                     #per fare ciò creo una maschera booleana che restituisce un array di true o false (più semplice)
    def band_power(low, high):
        idx= np.logical_and(freqs>=low, freqs<=high) #freqs>= low restituisce true se la freq è >= al limite inferiore
        return np.mean(psd[idx]) if np.any(idx) else 0 #restituisce la media aritmetica della potenza di banda mentre se nell'array risultasse tutto FALSE (if np.any(idx)) restituisce valore 0 evitando loop di errori
    def dominant_freq_in_band(low, high):
        idx=np.logical_and(freqs>=low , freqs<=high)
        if not np.any(idx) :
            return 0
        max_idx= np.argmax(psd[idx])#indice del max picco di potenza all'interno della porzione psd[idx] dell'array psd
        return freqs[idx][max_idx]#restituisce il valore di frequenza corrispondente
    alpha= band_power(8, 12)
    beta= band_power(13, 30)
    theta= band_power(4, 8)
    total_power= np.sum(psd)
    features= {"mean": float(np.mean(filtered)),
                "std": float(np.std(filtered)),
                 "alpha_power": float(alpha),
                 "beta_power": float(beta),
                 "theta_power": float(theta),
                 "beta_alpha_ratio": float(beta / alpha) if alpha > 0 else 0,
                 "dominant_alpha_freq": float(dominant_freq_in_band(8, 12)),
                 "dominant_beta_freq": float(dominant_freq_in_band(13, 30)),
                 "dominant_theta_freq": float(dominant_freq_in_band(4, 8)),
                 "total_power": float(total_power),}
    return features
def read_eeg_packet(ser, fs=250, duration=2):
    num_campioni=int(fs*duration)
    campioni=[]                                    #array vuoto
    while len(campioni)< num_campioni:             #ciclo per leggere via seriale e riempire la lista
        try :           #gestione errori
            line=ser.readline().decode(errors='ignore').strip() #pulisco i byte letti da spazi bianchi mettendoli in una riga
            if not line: 
                continue
            val=float(line) #converione in decimale
            campioni.append(val) #aggiungi linea alla lista
        except ValueError:
            print("errore conversione valore in float")
        except UnicodeDecodeError:
            print("errore nella decodifica dei caratteri")
    return np.array(campioni), fs   #restituisce array campioni e le frequenze
#====script principale====
def main():
    print(f"CONNETTENDO A {SERIAL_PORT}...")
    try:
       ser= serial.Serial(SERIAL_PORT, BAUD_RATE) #si prepara a leggere dalla seriale
    except Exception as e:
        print(f"errore apertura seriale: {e}")
        return
    time.sleep(2)
    print ("Seriale connessa con successo!")
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE)> 0:
        df=pd.read_csv(OUTPUT_FILE) #se il file esiste e ha dim.>0 lo carichiamo nel data frame
    else : df=pd.DataFrame() #se il file non esiste carichiamo tutto nel dataframe vuoto
    while True:
        input("Premi invio per registrare un pacchetto da due secondi")  
        eeg, fs= read_eeg_packet(ser, FS, DURATA) #acquiszione con la funzione read_eeg_packet usando le costanti assegnate da me (FS, DURATA), il rusuktato viene assegnato a eeg e fs
        print(f"{len(eeg)} campioni ricevuti , fs={fs}Hz")  #stampa la lunghezza dell'eeg e la frequenza
        feats=extract_features(eeg, fs)  #uso la funzione extract_feature e assegno le features estratte alla variabile feats
        print(f"features estratte: {feats}")

        etichetta=input("Inserisci un'etichetta per il pacchetto registrato (1--> concentrato e 0-->rilassato):")   
        try:
            feats["etichetta"]=int(etichetta)
        except ValueError:
            print("valore non riconosciuto assegno -1")
            feats["etichetta"]=-1
        df=pd.concat([df, pd.DataFrame([feats])], ignore_index=True)  #aggiungo le nuove feature al dataframe principale (df), pd.DataFrame([feats]) converte il dizionario in una riga apposita per il dataframe, pd.concat unisce al vecchio df la nuova riga e ignore_index resettaggiorna la numerazione delle righe
        df.to_csv( OUTPUT_FILE, index=False) #con index false elimino la colonna degli indici delle righe perche non mi serve nel train model
        print("✅Pacchetto salvato con successo in CSV!")
if __name__== "__main__":
    main()