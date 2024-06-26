clear all;
fs=16000;
c = 340;

T60r_a = 0.2;
T60r_b = 0.3;
Lrange_a = 3;
Lrange_b = 3.5;
thres_wall = 0.5;
thres_sm = 0.3;
thres_weights=0.25;
LL=[];
MM=[];
SS=[];
BB=[];
HH={};

T60_r = 0.2:0.01:2;
for t60 = T60_r
    Lrange_a = 1.5+exp(t60)-0.5;
    Lrange_b = 1.5+exp(t60)+0.5;
    Lrange = Lrange_a + (Lrange_b-Lrange_a).*rand(3,1);
    V = prod(Lrange);
    S = 2.*(Lrange(1).*Lrange(3) + Lrange(2).*Lrange(3) + Lrange(1).*Lrange(2));
    alfa_av = 24.*V.*log(10)./(c.*S.*t60);
    while alfa_av > 0.99
        Lrange = Lrange_a + (Lrange_b-Lrange_a).*rand(3,1);
        V = prod(Lrange);
        S = 2.*(Lrange(1).*Lrange(3) + Lrange(2).*Lrange(3) + Lrange(1).*Lrange(2));
        alfa_av = 24.*V.*log(10)./(c.*S.*t60);
    end
    L = [Lrange(1),Lrange(2),Lrange(3)];
    
    m = thres_wall + (repmat(L,[20,1])-thres_wall - thres_wall).*rand(20,3);
    s = thres_wall + (repmat(L,[5,1])-thres_wall - thres_wall).*rand(5,3);
    while any(distcalc(m,s) < thres_sm)
        m = thres_wall + (repmat(L,[20,1])-thres_wall - thres_wall).*rand(20,3);
        s = thres_wall + (repmat(L,[5,1])-thres_wall - thres_wall).*rand(5,3);
    end
    Ns = ceil(fs.*t60*1.2);
    beta_av = sqrt(1-alfa_av);
    weights = thres_weights + ((1-thres_weights) - thres_weights).*rand(2,1);
    w_v = [1-weights(1),weights(1),-1+weights(1),-weights(1),0.5,-0.5];
    w_v = w_v(randperm(length(w_v)));
    betas = 2.*w_v.*repmat(beta_av,[1,6]);
    while any(abs(betas) > 1)
        weights = thres_weights + ((1-thres_weights) - thres_weights).*rand(2,1);
        w_v = [1-weights(1),weights(1),-1+weights(1),-weights(1),0.5,-0.5];
        w_v = w_v(randperm(length(w_v)));
        betas = 2.*w_v.*repmat(beta_av,[1,6]);
    end
    h = rir_generator_x_threaded(c, fs, m, s, L, betas, Ns, 'omni', -1, [1,1,1], 0, false, true, 0.008);
    HH={HH;h};
    LL = [LL ; L];
    MM = [MM; m];
    SS=[SS;s];
    BB=[BB;betas];
end
save('./rirs_test.mat','-v7.3');

% plot_scenario(L,m,s)
% plot(h(:,1,1))
h_norm =h(:,1,1)./max(h(:,1,1));
n=find(h_norm==1)-1;
h_ns=zeros(size(h_norm));
h_ns(1:end-n)=h_norm(n+1:end);
figure(1);
clf;
plot(h_ns)
figure(2);
clf;
plot(20.*log10(abs(h_ns)))

% 
% h = rir_generator_x(c, fs, m, s, L, [1,1,1,1,1,1], Ns, 'omni', -1, [1,1,1], 0, false, true, 0.008);