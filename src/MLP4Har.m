function res = MLP4Har()
  
  f = fopen('XOR.txt');
  A = fscanf(f, '%g',[3 inf]);  
  A = A; 
  p = A(1:2, :)';%训练输入数据
  t = A(3, :)';%desire out
  [train_num , input_scale]= size(p) ;%规模
  fclose(f);
 
  alpha  = 0.1;%学习率
  moment_const = 0.9;%时刻因子
  threshold = 0.3;%  收敛条件 ∑e^2 < threshold
  wd1=0;  wd2=0; 
  bd1=0;  bd2=0;  	
  accumulate_error = 0.0; % 统计∑e^2
  circle_time =0;
  hidden_unitnum = 4; %隐藏层的单元数
  w1 = rand(hidden_unitnum,2);%4个神经元，每个神经元接受2个输入
  w2 = rand(1,hidden_unitnum);%一个神经元，每个神经元接受4个输入
  b1 = rand(hidden_unitnum,1);
  b2 = rand(1,1);

  while 1
    accumulate_error = 0.0;
    circle_time = circle_time +1;
	
    for i=1:train_num
        %前向传播
        a0 = double ( p(i,:)'  );%第i行数据
        n1 = w1*a0+b1;
        a1 = logsig(n1);%sigmoid(n1);%第一个的输出
        n2 = w2*a1+b2;
        a2 = logsig(n2);%sigmoid(n2);%第二个的输出
        a = a2;

        %后向传播敏感性
        e = t(i,:)-a;
        accumulate_error = accumulate_error + abs(e)^2;
                               %上面应该是平方吧 不平方错误依旧
        s2 = -2.*F(n2)*e; 	      
        s1 =    F(n1)*w2'*s2;

        %修改权值
        wd1 =  moment_const.*wd1 + alpha .* s1*a0';
        wd2 =  moment_const.*wd2 + alpha .* s2*a1';
        w1 =  w1 -wd1;
        w2 =  w2 -wd2;
        bd1 =  moment_const.*bd1 - alpha .* s1;
        bd2 =  moment_const.*bd2 - alpha .* s2;
        b1 = b1-bd1;
        b2 = b2-bd2;        
    end;%end of for
    if accumulate_error <= threshold| circle_time>10000  %then	
        	break;
    end;%end of if
  end;%end of while

disp(['accumulate_error = ',num2str( accumulate_error)] )	;
disp('------------');disp(circle_time )	

%----------------------------------------------------------
function [a]= sigmoid(n)
	a = 1./(1+exp(-n));
%----------------------------------------------------------
function [result]= F(a)
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = (1-a(i))*a(i);
	end;

