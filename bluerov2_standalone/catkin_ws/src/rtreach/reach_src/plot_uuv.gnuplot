set terminal png font arial 16 size 500, 600
set output "reach_uuv.png"

set title "Plot of Reachtubes, reachtime=2.0 s" font "arial,14"
set xlabel "x (m)"
set ylabel "y (m)"

# ranges for pendulum
set autoscale y
set xrange [-1958.106790:-1962.106790]
set yrange [41.406087:43.406087]

load "uuv_initial.gnuplot.txt"
load "uuv_final.gnuplot.txt"
load "uuv_intermediate.gnuplot.txt"


plot "uuv_simulation.dat" using 1:2 with point title ''
plot \
   1/0 lw 4 lc rgb 'blue' with lines t 'Initial', \
   1/0 lw 4 lc rgb 'dark-green' with lines t 'Intermediate', \
   1/0 lw 4 lc rgb 'red' with lines t 'Final'