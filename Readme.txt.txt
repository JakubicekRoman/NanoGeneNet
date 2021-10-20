4_0.pt - skript"LSTM_with_spectral.py" = sit s naucenim kátkých úseků 50tis, prevod na spektra 500 a nasledne LSTM, loss-WCE, ale obraceni, nevim proc
5_0.pt - upravena 4 a naucena, encoder misto spektra, lstm, 50tis vyrezy, vzdy s genem
5_1.pt - pokus, nove doucene 5_0 (jedna epocha) na celych signalech puvodni delky s enkoderovym podvzorkovanim
5_2.pt - preucena na 5_1 na dalsi signaly, nahodne vyrezy vzdy s genem, ale ruzne dlouhe
5_3.pt - naucene take na prazdnych signalech a o ruzne delce = prosty nahodny vyrez + nahodny scaling 5% delky - nefunguje
6_0.pt - naucit na signalech puvodni delky tzn. bez vyrezu, zda se bude sirit grad
7_0.pt - naucene z 5_0, kombinace vsech nacitacu, gen, rand i whole
