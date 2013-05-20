set size 1,2
set grid
set multiplot layout 2,1
set autoscale
set title "neural output"
unset key
plot "training.out" u 1 w lp pt 0.6 ps 0.5 
set autoscale
set title "target values"
unset key
plot "training.out" u 2 w lp pt 0.6 ps 0.5 
unset multiplot
set size 1,1
