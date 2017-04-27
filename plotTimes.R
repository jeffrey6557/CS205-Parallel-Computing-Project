plotTimes = function(seq, openMP, pycuda){
  plot(c(seq[1], openMP[1], pycuda[1]), pch = 16, col = 1:3, ylim = c(0, max(seq, openMP, pycuda)), 
      xaxt = "n", xlab = "", ylab = "Time in seconds", xlim = c(0.5,3.5), main = "Running time with \n95% CI" )
  axis(side = 1, at = 1:3, labels = c("Sequential", "OpenMP", "PyCuda"))
  segments(1,seq[2],1,seq[3])
  segments(2,openMP[2],2,openMP[3], col = 2)
  segments(3,pycuda[2],3,pycuda[3], col = 3)
  segments(1-.02,seq[2:3],1+.02,seq[2:3])
  segments(2-.02,openMP[2:3],2+.02,openMP[2:3], col = 2)
  segments(3-.02,pycuda[2:3],3+.02,pycuda[2:3], col = 3)
}


# Exercise 2 (Adam?)
seq = c(18.5330983472,11.0279892445, 30.1942003429)
openMP = c(14.9100168781, 9.60147128105, 26.312218219)
pycuda = c(7.50733185935, 4.52090799212, 13.1926374018)

png("images/adam_time.png")
plotTimes(seq, openMP, pycuda)
dev.off()

# Exercise 4 (Hessian-Free)
seq = c(18.5330983472,11.0279892445, 30.1942003429)
openMP = c(14.9100168781, 9.60147128105, 26.312218219)
pycuda = c(7.50733185935, 4.52090799212, 13.1926374018)

png("images/adam_time.png")
plotTimes(seq, openMP, pycuda)
dev.off()
