

# esegui e vedi differenze
clingo -n 0 --time-limit=60 --quiet=1,0,0 --opt-mode=optN original_encoding.lp input/days_1/input1.lp > as1.txt

clingo -n 0 --time-limit=60 --quiet=1,0,0 --opt-mode=optN optimized_encoding.lp input/days_1/input1.lp > as2.txt

diff -u as1.txt as2.txt



# numero di predicati (crescono linearmente col numero dei giorni)

clingo -n 0 input/days_1/input1.lp | grep -o 'count_[^ ]*'
    count_an(40)
    count_registration(70)
    count_mss(20)
    count_surgeon(20)
    count_surgeryTime(20)
    count_time(10)
    count_anaesthetistWT(20)

clingo -n 0 input/days_2/input1.lp | grep -o 'count_[^ ]*'
    count_an(80)
    count_registration(140)
    count_mss(40)
    count_surgeon(40)
    count_surgeryTime(40)
    count_time(20)
    count_anaesthetistWT(40)

clingo -n 0 input/days_3/input1.lp | grep -o 'count_[^ ]*'
    count_an(120)
    count_registration(210)
    count_mss(60)
    count_surgeon(60)
    count_surgeryTime(60)
    count_time(30)
    count_anaesthetistWT(60)

clingo -n 0 input/days_5/input1.lp | grep -o 'count_[^ ]*'
    count_an(200)
    count_anaesthetistWT(100)
    count_mss(100)
    count_registration(350)
    count_surgeon(100)
    count_surgeryTime(100)
    count_time(50)



