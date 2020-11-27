package com.passta.a2ndproj.start.activity;

import androidx.appcompat.app.AppCompatActivity;

import com.passta.a2ndproj.MainActivity;
import com.passta.a2ndproj.R;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

public class Page4Activity extends AppCompatActivity implements View.OnClickListener{
    private TextView next;
    private ImageView back;
    private TextView bank_data;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_page4);

        setStatusBar();
        InitializeView();
        SetListener();
    }


    private void setStatusBar() {
        View view = getWindow().getDecorView();
        view.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        getWindow().setStatusBarColor(Color.parseColor("#ffffff"));//색 지정

    }
    public void InitializeView()
    {

        next = (TextView)findViewById(R.id.next_page4_activity);
        back = (ImageView)findViewById(R.id.back_page4_activity);
        bank_data = (TextView)findViewById(R.id.bank_data);


    }

    public void SetListener(){
        next.setOnClickListener(this);
        back.setOnClickListener(this);
        bank_data.setOnClickListener(this);

    }

    public void onClick(View v){
        switch (v.getId()){
            case R.id.next_page4_activity:
                Intent intent1 = new Intent(getApplicationContext(), MainActivity.class);
                intent1.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK);
                startActivity(intent1);
                break;
            case R.id.back_page4_activity:
                goToNextActivity(new Page3Activity());
                break;
            case R.id.bank_data:
                Intent intent = new Intent(getApplicationContext(), SearchBankActivity.class);
//                    intent.putExtra("setting", "login");
                startActivity(intent);

                break;
        }
    }
    private void goToNextActivity(Activity activity) {
        finish();
        //intent 할때 앞 액티비티 스택을 다지우면서 가야함.
        Intent intent = new Intent(getApplicationContext(), activity.getClass());
        startActivity(intent);
    }
}