function res = MLP4Har()
  
  f = fopen('XOR.txt');
  A = fscanf(f, '%g',[3 inf]);  
  A = A; 
  p = A(1:2, :)';%ѵ����������
  t = A(3, :)';%desire out
  [train_num , input_scale]= size(p) ;%��ģ
  fclose(f);
 
  alpha  = 0.1;%ѧϰ��
  moment_const = 0.9;%ʱ������
  threshold = 0.3;%  �������� ��e^2 < threshold
  wd1=0;  wd2=0; 
  bd1=0;  bd2=0;  	
  accumulate_error = 0.0; % ͳ�ơ�e^2
  circle_time =0;
  hidden_unitnum = 4; %���ز�ĵ�Ԫ��
  w1 = rand(hidden_unitnum,2);%4����Ԫ��ÿ����Ԫ����2������
  w2 = rand(1,hidden_unitnum);%һ����Ԫ��ÿ����Ԫ����4������
  b1 = rand(hidden_unitnum,1);
  b2 = rand(1,1);

  while 1
    accumulate_error = 0.0;
    circle_time = circle_time +1;
	
    for i=1:train_num
        %ǰ�򴫲�
        a0 = double ( p(i,:)'  );%��i������
        n1 = w1*a0+b1;
        a1 = logsig(n1);%sigmoid(n1);%��һ�������
        n2 = w2*a1+b2;
        a2 = logsig(n2);%sigmoid(n2);%�ڶ��������
        a = a2;

        %���򴫲�������
        e = t(i,:)-a;
        accumulate_error = accumulate_error + abs(e)^2;
                               %����Ӧ����ƽ���� ��ƽ����������
        s2 = -2.*F(n2)*e; 	      
        s1 =    F(n1)*w2'*s2;

        %�޸�Ȩֵ
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

