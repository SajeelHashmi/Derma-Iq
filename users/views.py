from django.shortcuts import render , redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.core.validators import EmailValidator
from django.conf import settings
from django.contrib import messages
from predictor.models import Result

"""Basic auth complete add stripe subscription to register view and check subscription on login view"""

def login_view(request):
    if request.method == 'POST':
        if not request.user.is_authenticated:
            username = request.POST.get('username')
            password = request.POST.get('pass')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request=request,message="user logged in successfully")
                print("user logged in successfully")

                return redirect("home")
            else:
                messages.error(request=request,message="incorrect user name or password")
                return redirect('login')
        else:
            return render(request , 'login.html', {})
    return render( request, 'login.html' )

def logout_user(request):
    logout(request=request)
    return redirect("login")

    
def sign_up(request):
    if request.user.is_authenticated:
        return redirect("home")
    if request.method =="POST":

        # print("method post")
        username = request.POST["username"]
        pass1 = request.POST["passowrd"]
        pass2 = request.POST["confirm_password"]

        # validate the data in the form
        if pass1==pass2:
            user = User.objects.create_user(username=username, password=pass1)
            user.save()
            login(request,user)
            return redirect("home")
        else:
            messages.success(request=request,message="PASSWORDS DONOT MATCH")
    return render(request,"signup.html")



def home(request):
    if request.user.is_authenticated:
        results = Result.objects.filter(user=request.user).order_by('-created_at')

        return render(request, 'home.html', {"results": results})
    else:
        return redirect('login')