m = memmapfile('/Users/zhd/Desktop/Research/Project/IARPA data/first_try/M.mmap','Format',{'double',[5000 61],'V_mat'});
V_mat = m.Data.V_mat;
n_comp = size(V_mat,2);
[rlt,rlt2] = FastSepNMF_dz(V_mat,n_comp,1,0.6);
save('spa_rlt8.mat','rlt');
