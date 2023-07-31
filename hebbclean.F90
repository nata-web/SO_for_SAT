module hebb
  ! compile with:
  ! f2py3 --f90flags="-g -fdefault-integer-8 -O3" -m hebbF -c hebbclean.F90
  implicit none
CONTAINS
  SUBROUTINE runsimple(w,I,c,worig,Iorig,corig,energies,steps,repeats,n,dolearn,alpha,state,randoms)
    IMPLICIT NONE

    ! Eend means only compute the energy at the end of the run

    ! w, worig, I,c need to be pre-multiplied by alpha

    ! returns energies and state
    ! the energies need to be divided by alpha afterwards

    INTEGER*8, INTENT(IN) :: steps,repeats,n,dolearn,randoms(steps+n,repeats)
    real*8, INTENT(INOUT) :: w(n,n)
    real*8 , INTENT(IN) :: I(n),c,worig(n,n),Iorig(n),corig,alpha
    real*8, DIMENSION(steps,repeats),intent(inout) :: energies
    integer*1,intent(inout) :: state(n)

    integer*8 :: j
    do j=1,repeats
      call learnSimple(w,I,c,worig,Iorig,corig,steps,n,randoms(:,j),energies(:,j),dolearn,alpha,state)
      if (mod(j,(repeats/10))==0) then
        write (*,*) j
      end if
    end do
  end SUBROUTINE runsimple
  SUBROUTINE learnSimple(w,I_c,c,worig,I_corig,corig,steps,n,randoms,energies,dolearn,alpha,state)
    IMPLICIT NONE

    INTEGER*8 , INTENT(IN) :: steps,n,dolearn,randoms(steps+n)
    real*8 , INTENT(INOUT) :: w(n,n)
    real*8 , INTENT(IN) :: I_c(n),c,worig(n,n),I_corig(n),corig,alpha
    real*8, DIMENSION(steps), INTENT(inOUT) :: energies
    integer*1,intent(inout) :: state(n)

    integer*8 :: t,i,j,idx,delta
    real*8 :: activation
    real*8 :: r
    integer*8 ::oldState,newState
    integer*4,dimension(n) :: idx2t
    integer*4,dimension(steps) :: t2Idx
    integer*1,dimension(steps) :: t2State
    integer*1,dimension(n,n) :: dw
    idx2t(:)=1
    state(1:n)=randoms(1:n)
    dw(:,:)=0
    do t=1,steps
      idx=randoms(n+t)
      oldState=state(idx)
      if (dolearn==1) then
        w(:,idx)=w(:,idx)+dw(:,idx)*(t-idx2t(idx))*alpha
        do i=idx2t(idx)+1,t-1
          newState=state(idx)*t2State(i)
          if (newState/=dw(t2idx(i),idx)) then
            w(t2idx(i),idx)=w(t2idx(i),idx)+(newState-dw(t2idx(i),idx))*(t-i)*alpha
            dw(t2idx(i),idx)=newState
          end if
        end do
        idx2t(idx)=t
        t2idx(t)=idx
      end if
      activation=I_c(idx)+sum(w(:,idx)*state(:))
      if (activation>=0) then
        state(idx)=1
      else
        state(idx)=-1
      end if
      if (dolearn==1) then
        t2State(t)=state(idx)
        if (t==1) then
          do j=1,n
            dw(:,j)=state(:)*state(j)
          end do
        else
          if (state(idx)>0) then
            dw(:,idx)=state(:)
          else
            dw(:,idx)=-state(:)
          end if
        end if
      end if
      if (t==1) then
        ! state=[a,b,c,d,e]
        ! state(j)*state(k)*worig(j,k)
        ! e(t-1)=sum(
        ! [aa, ab, ac, ad, ae]
        ! [ba, bb, bc, bd, be]
        ! [ca, cb, cc, cd, ce]
        ! [da, db, dc, dd, de]
        ! [ea, eb, ec, ed, ee]
        ! )

        ! e(t)=sum(
        ! [aa, aB, ac, ad, ae]
        ! [Ba, BB, Bc, Bd, Be]
        ! [ca, cB, cc, cd, ce]
        ! [da, dB, dc, dd, de]
        ! [ea, eB, ec, ed, ee]
        ! )

        ! e(t)=e(t-1)
        ! +2*(state(idx)-oldState(idx))*sum_{j=1..n,j/=idx} state(j)*worig(idx,j)
       
        ! energies(t)= state^T worig state
        ! energies(t)=0
        ! energies(t)=sum_{j=1..n,k=1..n} (state(j)*state(k)*worig(j,k))
        ! energies(t)=sum_{j=1..n,k=1..n} (state(k)*state(j)*worig(k,j))
        ! energies(t)=sum_{j=1..n} state(j)*state(j)*worig(j,j)
        !                          +2*sum_{k=j+1..n} (state(k)*state(j)*worig(k,j))
        energies(t)=-sum(state(:)*I_cOrig(:))-cOrig
        do j=1,n
          ! wOrig is symmetric, and state(j)**2=1
          !energies(t)=energies(t)-state(j)*sum(state(:)*worig(:,j))
          energies(t)=energies(t)&
               -worig(j,j)/2.0d0&
               -state(j)*sum(state(1:j-1)*worig(1:j-1,j))
        end do
      else
        if (state(idx)/=oldState) then
          ! only state(idx) changes, so I can update the last
          ! energy by the change
          energies(t)=energies(t-1)&
               -(state(idx)-oldState)&
               *(sum(state(1:idx-1)*worig(1:idx-1,idx))&
               +sum(state(idx+1:n)*worig(idx+1:n,idx))+I_cOrig(idx))
          ! this term is zero, runsimple is bipolar only so state(idx)**2==oldState**2
          ! energies(t)=energies(t)-(state(idx)**2-oldState**2)*w(idx,idx)*0.5d0
        else
          energies(t)=energies(t-1)
        end  if
      end if
    end do
    if (dolearn==1) then
      t=steps+1
      do idx=1,n
        w(:,idx)=w(:,idx)+dw(:,idx)*(t-idx2t(idx))*alpha
        do i=idx2t(idx)+1,t-1
          newState=state(idx)*t2State(i)
          if (newState/=dw(t2idx(i),idx)) then
            w(t2idx(i),idx)=w(t2idx(i),idx)+(newState-dw(t2idx(i),idx))*(t-i)*alpha
            dw(t2idx(i),idx)=newState
          end if
        end do
      end do
    end if
  end SUBROUTINE learnSimple
end module hebb
