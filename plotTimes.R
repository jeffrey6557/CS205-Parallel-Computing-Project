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

# Build colors
sampCol = sample(colors(), length(times))

# Figure 1 (time until convergence)
hess = c(36.7705330849, 40.8134579659, 30.7013020515, 30.7084610462, 27.9912378788, 32.4583690166, 32.423830986, 32.7279911041, 29.1763651371, 30.2343609333, 31.1755859852, 32.7965919971, 27.9905948639, 29.830283165, 30.0658519268, 36.020111084, 29.293243885, 29.3104069233, 29.4974980354, 32.6336672306, 31.5671169758, 31.8110649586, 44.7567741871, 35.8124980927, 29.3117141724, 29.445168972, 36.4843299389, 27.869538784, 37.7073988914 , 36.1895840168, 27.562016964, 26.9184501171, 30.9687809944, 27.565570116, 26.0636160374, 29.935836792, 29.1046249866, 28.483631134, 30.4000070095, 29.424945116, 47.326941967, 27.9466600418, 31.4740049839, 23.681566, 29.9285681248, 28.4917099476, 25.6063570976, 31.7247228622, 27.9845449924, 42.1240811348, 26.9783420563, 29.4615120888, 38.4782221317, 30.3626039028, 30.5635900497, 29.5388019085, 35.8142380714, 29.1841518879, 44.0196020603, 35.7132070065, 31.4251890182, 27.636906147, 37.33061409, 29.5162479877, 27.6029720306, 27.6011309624, 34.7988889217, 30.9964210987, 30.7129850388, 27.3245060444, 29.6513118744, 35.2752830982, 32.2567999363, 28.2544419765, 29.1040530205, 27.9447622299, 33.2662549019, 30.2134070396, 31.2737631798, 32.4858360291, 29.9900941849, 32.8325059414, 34.9779760838, 34.1400580406, 26.3196659088, 33.1836740971, 34.5855379105, 42.6495320797, 28.9761240482, 29.3485779762, 37.6159031391, 42.8580520153, 27.6141171455, 29.2168259621, 27.0530879498, 32.8354289532, 27.9998650551, 36.4524850845, 33.5226078033, 31.4912500381)
hess = mean(hess)
ada_gpu = 102.309973001
ada_cpu = 117.10699296
mpi3n4c = 128.506565809
mpi4n4c = 71.5182712078
mpi4n8c = 57.5800118446
mpi5n4c = 54.687912941
mpi6n4c = 58.7733278275
mpi7n4c = 68.8883149624
mpi8n4c = 127.44204998
big_gpu = 128.102043867 #75
big_cpu = 168.119750977 #75
big3n4c = 165.719032049 #76
big4n4c = 95.7436389923 #72
big4n8c = 421.072537184 #471
big5n4c = 90.6643800735 #76
big6n4c = 94.1150431633 #45
big7n4c = 304.533075809 #531
big8n4c = 103.547483921 #131
times = c(hess, ada_cpu,ada_gpu,mpi3n4c,mpi4n4c,mpi4n8c,mpi5n4c,mpi6n4c,mpi7n4c,mpi8n4c, big_cpu, big_gpu, big3n4c, big4n4c, big4n8c, big5n4c, big6n4c, big7n4c, big8n4c)

png("images/plot1_timeUntilConvergence.png")
par(mar=c(1, 4.1, 4.1,10.1), xpd=TRUE)
barplot(times, col = sampCol, main = "Running time", ylab = "Seconds")
legend(length(times)+5.5,400, legend = c("Hessian-free GPU", "ADA CPU","ADA GPU","MPI+ADA 3 nodes 4 cores", "MPI+ADA 4 nodes 4 cores", "MPI+ADA 4 nodes 8 cores", "MPI+ADA 5 Nodes 4 cores", "MPI+ADA 6 nodes 4 cores", "MPI+ADA 7 nodes 4 cores", "ADA CPU\nbig batch","ADA GPU\nbig batch","MPI+ADA 3 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 8 cores\nbig batch", "MPI+ADA 5 Nodes 4 cores\nbig batch", "MPI+ADA 6 nodes 4 cores\nbig batch", "MPI+ADA 7 nodes 4 cores\nbig batch"), fill = sampCol, cex = .6, y.intersp = 1.5)
dev.off()

# Figure 2 (time with fixed number of intervals)
hess = 5669.19178391/1000
ada_cpu = 102.309973001/11
ada_gpu = 117.10699296/40
mpi3n4c = 128.506565809/50
mpi4n4c = 71.5182712078/41
mpi4n8c = 57.5800118446/71
mpi5n4c = 54.687912941/32
mpi6n4c = 58.7733278275/43
mpi7n4c = 68.8883149624/54
mpi8n4c = 127.44204998/155
big_gpu = 128.102043867/75
big_cpu = 168.119750977/75
big3n4c = 165.719032049/76
big4n4c = 95.7436389923/72
big4n8c = 421.072537184/471
big5n4c = 90.6643800735/76
big6n4c = 94.1150431633/45
big7n4c = 304.533075809/531
big8n4c = 103.547483921/131
times = c(hess, ada_cpu,ada_gpu,mpi3n4c,mpi4n4c,mpi4n8c,mpi5n4c,mpi6n4c,mpi7n4c,mpi8n4c, big_cpu, big_gpu, big3n4c, big4n4c, big4n8c, big5n4c, big6n4c, big7n4c, big8n4c)


png("images/plot2_timeWithFixedIterations.png")
par(mar=c(1, 4.1, 4.1,10.1), xpd=TRUE)
barplot(times, col = sampCol, main = "Running time per iteration", ylab = "Seconds")
legend(length(times)+5.5,9, legend = c("Hessian-free GPU", "ADA CPU","ADA GPU","MPI+ADA 3 nodes 4 cores", "MPI+ADA 4 nodes 4 cores", "MPI+ADA 4 nodes 8 cores", "MPI+ADA 5 Nodes 4 cores", "MPI+ADA 6 nodes 4 cores", "MPI+ADA 7 nodes 4 cores", "ADA CPU\nbig batch","ADA GPU\nbig batch","MPI+ADA 3 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 8 cores\nbig batch", "MPI+ADA 5 Nodes 4 cores\nbig batch", "MPI+ADA 6 nodes 4 cores\nbig batch", "MPI+ADA 7 nodes 4 cores\nbig batch"), fill = sampCol, cex = .6, y.intersp = 1.5)
dev.off()

## Plots per node
# Total time
n = c(3:8)
ada_n = c(128.506565809, 71.5182712078, 54.687912941, 58.7733278275, 68.8883149624, 127.44204998)
big_n = c(165.719032049,95.7436389923,90.6643800735, 94.1150431633, 304.533075809, 103.547483921)
big20 = c(462.547384977,1117.19513893,768.999130011,721.323318958,635.223259926,630.516092062)
ada20 = c(1228.72130013,906.030404091,823.905865192,635.600432873, 605.052289009,444.832736969)

png("images/plot3_ADAperNode_time.png")
plot(n, ada_n, type = "l", col = "green", lwd = 3, ylim = c(0, max(ada_n, big_n, ada20, big20)), xlab = "Number of Nodes", ylab = "Time in seconds (per iteration)", main = "Runtime of ADA \nOptimization with 4 cores")
lines(n, big_n, col = "dark green", lwd = 3)
lines(n, ada20, col = "green", lty = 3, lwd = 3)
lines(n, big20, col = "dark green", lty = 3, lwd = 3)
legend("topright", legend = c("1024 with chain length 1", "4096 with chain length 1", "1024 with chain length 20", "4096 with chain length 20"), col = rep(c("green", "dark green"),2), lwd = 3, lty = c(1,1,3,3), title = "Batch size", cex = .7)
dev.off()

# Time per iteration
ada_n = c(128.506565809/50, 71.5182712078/41, 54.687912941/32, 58.7733278275/43, 68.8883149624/54, 127.44204998/155)
big_n = c(165.719032049/76, 95.7436389923/72, 90.6643800735/76, 94.1150431633/45, 304.533075809/531, 103.547483921/131)
ada20 = c(1228.72130013/98,906.030404091/100,823.905865192/100,635.600432873/100,605.052289009/100, 444.832736969/100)
big20 = c(462.547384977/27/20,1117.19513893/100/20,768.999130011/80/20,721.323318958/100100/20,635.223259926/100/20,630.516092062/100/20)
png("images/plot4_ADAperNode_timePerIteration.png")
plot(n, ada_n, type = "l", col = "green", lwd = 3, ylim = c(0, max(ada_n, big_n, ada20, big20)), xlab = "Number of Nodes", ylab = "Time in seconds (per iteration)", main = "Runtime per Iteration of ADA \nOptimization with 4 cores")
lines(n, big_n, col = "dark green", lwd = 3)
lines(n, ada20, col = "green", lty = 3, lwd = 3)
lines(n, big20, col = "dark green", lty = 3, lwd = 3)
legend("topright", legend = c("1024 with chain length 1", "4096 with chain length 1", "1024 with chain length 20", "4096 with chain length 20"), col = rep(c("green", "dark green"),2), lwd = 3, lty = c(1,1,3,3), title = "Batch size", cex = .7)
dev.off()

## Accuracy:
hess = 0.529322816625
ada_gpu = 0.583682435884
ada_cpu = 0.44364973945
mpi3n4c = 0.578675794421
mpi4n4c = 0.570246245019
mpi4n8c = 0.421681822826
mpi5n4c = 0.42014917748
mpi6n4c = 0.450546643507
mpi7n4c = 0.584397670379
mpi8n4c = 0.584857463983
big_gpu = 0.450904260754 #75
big_cpu = 0.450904260754 #75
big3n4c = 0.415857770512 #76
big4n4c = 0.561663431082 #72
big4n8c = 0.416368652294 #471
big5n4c = 0.576070297333 #76
big6n4c = 0.41534688873 #45
big7n4c = 0.416573005007 #531
big8n4c = 0.576325738224 #131
times = c(hess, ada_cpu,ada_gpu,mpi3n4c,mpi4n4c,mpi4n8c,mpi5n4c,mpi6n4c,mpi7n4c,mpi8n4c, big_cpu, big_gpu, big3n4c, big4n4c, big4n8c, big5n4c, big6n4c, big7n4c, big8n4c)

png("images/plot5_accuracy.png")
par(mar=c(1, 4.1, 4.1,10.1), xpd=TRUE)
barplot(times, col = sampCol, main = "Model Accuracies", ylab = "Accuracy %", ylim = c(.4,.6), xpd = F)
legend(length(times)+5.5,.6, legend = c("Hessian-free GPU", "ADA CPU","ADA GPU","MPI+ADA 3 nodes 4 cores", "MPI+ADA 4 nodes 4 cores", "MPI+ADA 4 nodes 8 cores", "MPI+ADA 5 Nodes 4 cores", "MPI+ADA 6 nodes 4 cores", "MPI+ADA 7 nodes 4 cores", "ADA CPU\nbig batch","ADA GPU\nbig batch","MPI+ADA 3 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 4 cores\nbig batch", "MPI+ADA 4 nodes 8 cores\nbig batch", "MPI+ADA 5 Nodes 4 cores\nbig batch", "MPI+ADA 6 nodes 4 cores\nbig batch", "MPI+ADA 7 nodes 4 cores\nbig batch"), fill = sampCol, cex = .6, y.intersp = 1.5)
dev.off()

# Chain 1 vs Chain 2:

