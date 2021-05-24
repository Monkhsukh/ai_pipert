CC = g++
CFLAGS = -std=c++11

test: main.o hung.o
	$(CC) -o test main.o hung.o

hung.o: HungarianAlgorithm.cpp HungarianAlgorithm.h
	$(CC) -c HungarianAlgorithm.cpp -o hung.o
	
main.o: testMain.cpp HungarianAlgorithm.h
	$(CC) $(CFLAGS) -c testMain.cpp -o main.o

clean:
	-rm main.o hung.o