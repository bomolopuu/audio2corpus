# audio2corpus
App to transcribe audio 2 text for any given language


We fine-tuned Meta AI’s large speech model 
`Massively Multilingual Speech (MMS) 1B [3][4]`
1,100+ languages supported in the model
Model was pre-trained on 100,000s hours of open source speech data 
MMS has adapters: bridges between the model’s existing knowledge and new specialized tasks

DATA
494 training phrases, 167 test phrases: audios, pre-processed transcriptions and translations
1,5 and 0,5 hours of speech data from 2 native speakers
FINE-TUNING
Adapter used for Ngen’s closest related language Bamana
Internal layers frozen
Parameters taken from related literature

