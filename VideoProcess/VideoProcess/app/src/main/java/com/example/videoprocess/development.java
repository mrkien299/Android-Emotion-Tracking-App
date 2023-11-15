package com.example.videoprocess;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class development extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_development);
    }
    public void switchActivity(View view) {
        Intent intent = new Intent(this, home_activity.class);
        startActivity(intent);
    }

}