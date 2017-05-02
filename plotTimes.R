plotTimes = function(data, labels){
  plot(data[1,], pch = 16, col = 1:ncol(data), ylim = c(0, max(data)), 
       xaxt = "n", xlab = "", ylab = "Time in seconds", xlim = c(0.5,ncol(data)+.5), 
       main = "Average Running Time with 95% CI" )
  axis(side = 1, at = 1:ncol(data), labels = labels, tick = F, las = 2)
  for (i in 1:ncol(data)){
    if (i < ncol(data)){abline(v = i+.5, lty = 2)}
    segments(i,data[2,i],i,data[3,i], col = i)
    segments(i-.02,data[2:3,i],i+.02,data[2:3,i], col = i)
  }
}

# Figure 1 (time until convergence)

hess = c(36.7705330849, 40.8134579659, 30.7013020515, 30.7084610462, 27.9912378788, 32.4583690166, 32.423830986, 32.7279911041, 29.1763651371, 30.2343609333, 31.1755859852, 32.7965919971, 27.9905948639, 29.830283165, 30.0658519268, 36.020111084, 29.293243885, 29.3104069233, 29.4974980354, 32.6336672306, 31.5671169758, 31.8110649586, 44.7567741871, 35.8124980927, 29.3117141724, 29.445168972, 36.4843299389, 27.869538784, 37.7073988914 , 36.1895840168, 27.562016964, 26.9184501171, 30.9687809944, 27.565570116, 26.0636160374, 29.935836792, 29.1046249866, 28.483631134, 30.4000070095, 29.424945116, 47.326941967, 27.9466600418, 31.4740049839, 23.681566, 29.9285681248, 28.4917099476, 25.6063570976, 31.7247228622, 27.9845449924, 42.1240811348, 26.9783420563, 29.4615120888, 38.4782221317, 30.3626039028, 30.5635900497, 29.5388019085, 35.8142380714, 29.1841518879, 44.0196020603, 35.7132070065, 31.4251890182, 27.636906147, 37.33061409, 29.5162479877, 27.6029720306, 27.6011309624, 34.7988889217, 30.9964210987, 30.7129850388, 27.3245060444, 29.6513118744, 35.2752830982, 32.2567999363, 28.2544419765, 29.1040530205, 27.9447622299, 33.2662549019, 30.2134070396, 31.2737631798, 32.4858360291, 29.9900941849, 32.8325059414, 34.9779760838, 34.1400580406, 26.3196659088, 33.1836740971, 34.5855379105, 42.6495320797, 28.9761240482, 29.3485779762, 37.6159031391, 42.8580520153, 27.6141171455, 29.2168259621, 27.0530879498, 32.8354289532, 27.9998650551, 36.4524850845, 33.5226078033, 31.4912500381)
hess = mean(hess)
ada_cpu = 102.309973001
ada_gpu = 117.10699296
mpi3n4c = 128.506565809
mpi4n4c = 71.5182712078
mpi4n8c = 57.5800118446
mpi5n4c = 54.687912941
mpi6n4c = 58.7733278275
mpi7n4c = 68.8883149624
mpi8n4c = 127.44204998
times = c(hess,ada_cpu,ada_gpu,mpi3n4c,mpi4n4c,mpi4n8c,mpi5n4c,mpi6n4c,mpi7n4c,mpi8n4c)

png("images/plot1_timeUntilConvergence.png")
par(mar=c(1, 4.1, 4.1,10), xpd=TRUE)
barplot(times, col = rainbow(length(times)), main = "Running time", ylab = "Seconds")
legend(12.5,100, legend = c("Hessian-free GPU", "ADA CPU","ADA GPU","MPI+ADA 3 nodes 4 cores", "MPI+ADA 4 nodes 4 cores", "MPI+ADA 4 nodes 8 cores", "MPI+ADA 5 Nodes 4 cores", "MPI+ADA 6 nodes 4 cores", "MPI+ADA 7 nodes 4 cores"), fill = rainbow(length(times)), cex = .7)
dev.off()

# Figure 2 (time with fixed number of intervals)
hess = 5669.19178391*2
ada_cpu = 102.309973001
ada_gpu = 117.10699296
mpi3n4c = 128.506565809
mpi4n4c = 71.5182712078
mpi4n8c = 57.5800118446
mpi5n4c = 54.687912941
mpi6n4c = 58.7733278275
mpi7n4c = 68.8883149624
mpi8n4c = 127.44204998
times = c(hess,ada_cpu,ada_gpu,mpi3n4c,mpi4n4c,mpi4n8c,mpi5n4c,mpi6n4c,mpi7n4c,mpi8n4c)

png("images/plot2_timeWithFixedIterations.png")
par(mar=c(1, 4.1, 4.1,10), xpd=TRUE)
barplot(times, col = rainbow(length(times)), main = "Running time", ylab = "Seconds")
legend(12.5,100, legend = c("Hessian-free GPU", "ADA CPU","ADA GPU","MPI+ADA 3 nodes 4 cores", "MPI+ADA 4 nodes 4 cores", "MPI+ADA 4 nodes 8 cores", "MPI+ADA 5 Nodes 4 cores", "MPI+ADA 6 nodes 4 cores", "MPI+ADA 7 nodes 4 cores"), fill = rainbow(length(times)), cex = .7)
dev.off()
