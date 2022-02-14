#include<stdio.h>
#include<stdlib.h>

int main(){
    int counter = 1000;
    int hour, minute, second;
    char* time_string = malloc(9);
    char* hstring;
    char* mstring;
    char* sstring;
    int counter = 123412412;

    int hour = counter / 3600;
    int minute = (counter - hour * 3600) / 60;
    int s = counter - hour * 3600 - minute * 60;
    time_string[0] = '0' + hour / 10;
    time_string[1] = '0' + hour % 10;
    time_string[2] = ':';
    time_string[3] = '0' + minute / 10;
    time_string[4] = '0' + minute % 10;
    time_string[5] = ':';
    time_string[6] = '0' + second / 10;
    time_string[7] = '0' + second % 10;
    time_string[8] = '\0';
}