# Translacja-jezyka-migowego

W celu poprawy danych wejściowych skorzystano z zewnętrznej biblioteki MediaPipe pozwalającej na śledzenie dłoni i wyekstrahowanie z obrazu sylwetki postaci.
![image](https://user-images.githubusercontent.com/67105405/213808018-eea2d6ba-be1d-47b9-a597-5d67ce4c8675.png)

Orientacyjne punkty charakteryzowane są przez 4 wartości, widoczność oraz pozycje na
osiach: x, y, z. W skład wektora danych wchodzi jedynie informacja o wartości na osiach x i
y. Pozycja na osi z została odrzucona z względu na dużą niedokładność. Skutkowało to uczeniem
się przez sieć neuronową z niewartościowych danych, zwiększając szanse na przeuczenie. Widocz-
ność służy do podjęcia decyzji o przypisaniu wartości do zmiennej lub zainicjowania jej zerem.

W wektor danych wchodzą punkty na rękach i ramionach (od 11 do 24), pozostałe punkty
nie są istotne dla predykcji. Punkty o parzystej pozycji w tablicy były przypisywane do zmiennej
określającej prawą rękę, natomiast nieparzyste do zmiennej określającej lewą rękę.

Punkty orientacyjne na głowie (od 0 do 10) zostały skoncentrowane do jednego punktu będą-
cym odniesieniem dla całego układu. Wartości x i y są liczone względem krawędzi obrazu, lewy
górny róg (0, 0), prawy dolny róg (100, 100). Obliczana jest różnica między każdym punktem
orientacyjnym, a wartością skoncentrowanego punktu. Dzięki tej operacji ruch kamery bądź na-
granie osoby porozumiewającej się językiem migowym w różnym kadrze nie wpływa na wartości
na osiach x i y.

![image](https://user-images.githubusercontent.com/67105405/213808510-057b1079-2239-438a-819a-d3ae4b94b180.png)

Do wektora danych wchodzą wszystkie punkty orientacyjne. Uwzględniane są tylko wartości
na osiach x i y (jak powyżej). W przeciwieństwie do poprzedniej tablicy nie istnieje parametr
widoczność dla poszczególnych punktów. W przypadku braku detekcji dłoni przypisywana jest
wartość pustej tablicy.

Wartość na osi x dla prawej dłoni oraz parzyste wartości w tablicy pochodzących z predykcji
pozycji ciała, odpowiadającej prawej ręce, zostały pomnożone przez wartość -1. Dzięki tej operacji
przestaje być konieczne tworzenie danych uczących osobno dla prawej i lewej ręki.

## Zbieranie danych dla 3 różnych sieci głębokiego uczenia maszynowego

Motywacja: użycie jednej sieci neuronowej natrafiło na szereg trudności. Mimo ogranicze-
nia wektora danych do istotnych informacji, zbiór danych uczących był zbyt mały by w spo-
sób zadowalający wytrenować sieć neuronową do klasyfikacji 20 znaków. W przypadku znaków
prezentowanych przez ruch tylko jednej ręki dane pochodzące z drugiej utrudniały prawidłową
klasyfikację. Dane były zainicjowane przypadkowi wartościami pochodzącymi z poprzedniej pre-
dykcji lub zerami. Podczas tworzenia nagrania nie jest możliwe nałożenie punktów orientacyjnych
na każdą klatkę z użyciem biblioteki MediaPipe. Dodatkowo nie zawsze na nagraniu widoczne
są obie ręce. Konsekwencją było istnienie zer w wektorze danych. Sieć neuronowa ulegała prze-
uczeniu utożsamiając istnienie zer na poszczególnych pozycjach w sekwencji z znakiem. Prostym
rozwiązaniem tego problemu mogłoby być wymuszenie obecności obu rąk na klatce i nałożenie
na nie punktów orientacyjnych. Ograniczyłoby to jednak funkcjonalność aplikacji. Użytkownik
zmuszony byłby do stworzenia sztucznych warunków, obecności 2 rąk na nagraniu nawet w sy-
tuacji, gdy znak jest wykonywany jest jedną ręką. Dodatkowo stworzenie nagrania złożonego z
samych klatek na których udałoby się nałożyć punkty orientacyjne jest niemal niemożliwe.

Z wyżej wymienionych powodów zdecydowano się podzielić znaki na 3 różne kategorie i do-
konywać predykcji dedykowanymi do tego sieciami neuronowymi. Stworzono zbiory danych dla
znaków wykonywanych jedną ręką, znaków wykonywanych dwoma rękami oraz znaków cechują-
cymi się znikomą zmiennością.

Podczas tworzenia sekwencji zliczana była ilość klatek z nie nałożonymi punktami orientacyjnymi. Dla ręki sprawdzano, istnienie zer w wektorze natomiast dla dłoni sprawdzano, czy tablica jest pusta, następnie inicjowano ją pustymi znakami. Po zebraniu 30 klatek wartości równe zero były zamieniane na wartość średnią z poprzedzającej i następującej po niej klatce. wariancja obliczana była z całej poprawionej sekwencji. były zamieniane na wartość średnią z poprzedzającej i następującej po niej klatce. wariancja obliczana była z całej poprawionej sekwencji.

##Rozwiązanie z użyciem rekurencyjnych sieci neuronowych

Do trenowania architektur sieci dla tego rodzaju rozwiązania zostały wykorzystane dane
zebrane za pomocą biblioteki MediaPipe. Ze względu na dokonany podział znaków na trzy pod-
zbiory: znaki o niskiej wariancji, znaki wykonywane jedną ręką oraz znaki wykonywane obiema
rękami do ostatecznego rozwiązania został wybrany jeden model dla każdego z nich. Do tre-
nowania każdej z wymienionych w tym podrozdziale sieci została zastosowana funkcja reakcji
EarlyStopping z biblioteki keras dzięki czemu uniknięto przetrenowania modelu.

### Znaki o niskiej wariancji
Ze względu na niesekwencyjny charakter danych został zastosowany prosty model sieci neu-
ronowej składający się z warstwy wejściowej, jednej warstwy ukrytej oraz warstwy wyjściowej.
![image](https://user-images.githubusercontent.com/67105405/213810393-a5229d5c-136a-417b-a483-6168905290de.png)
Do modelu zostały dobrane funkcja straty – kategoryczna entropia krzyżowa, optymalizator
– Adam, metryka wydajności – dokładność. Trenowanie modelu zakończyło się po 225 epokach,
osiągając następujące wyniki dokładność na poziomie 98% oraz wartość funkcji straty – 0.34.
Rysunek 5.5 przedstawia przebieg zmian wartości funkcji straty oraz dokładności podczas procesu
uczenia modelu.

### Modele dla danych sekwencyjnych


Do trenowania modelu dla znaków wykonywanych jedną oraz obiema rękoma została wykorzystana następująca architektury rekurencyjnych sieci neuronowych.
Sieć rekurencyjna składająca się z: warstwy wejściowej, warstwy wyjściowej, dwukierunkowych warstw komórek LSTM. Architektura sieci została przedstawiona na rysunku 5.9.
